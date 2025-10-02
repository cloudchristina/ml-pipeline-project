import re
from contextlib import contextmanager
from typing import Generator, List, Dict, Any, Optional

from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base
from ..utils.config import Config
from ..utils.logger import get_logger
from ..utils.exceptions import DatabaseError, SecurityError

logger = get_logger(__name__)


class DatabaseManager:
    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        self.SessionLocal = None
        self._valid_table_names = set()
        self._initialize_database()

    def _validate_table_name(self, table_name: str) -> str:
        """Validate table name to prevent SQL injection."""
        # Initialize valid table names if not already done
        if not self._valid_table_names and self.engine:
            metadata = MetaData()
            metadata.reflect(bind=self.engine)
            self._valid_table_names = set(metadata.tables.keys())
            # Also add our model table names
            for table_class in Base.__subclasses__():
                self._valid_table_names.add(table_class.__tablename__)

        # Check if table name contains only valid characters
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise SecurityError(
                f"Invalid table name format: {table_name}",
                {"table_name": table_name, "reason": "invalid_characters"}
            )

        # Check if table name is in our whitelist
        if table_name not in self._valid_table_names:
            raise SecurityError(
                f"Table name not allowed: {table_name}",
                {"table_name": table_name, "valid_tables": list(self._valid_table_names)}
            )

        return table_name

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            logger.info("Initializing database connection")

            # Create engine with connection pooling
            self.engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=5,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL debugging
            )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # Create tables
            self.create_tables()

            logger.info("Database initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def create_tables(self):
        """Create all tables if they don't exist."""
        try:
            logger.info("Creating database tables")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")

        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise

    def drop_tables(self):
        """Drop all tables - USE WITH CAUTION."""
        try:
            logger.warning("Dropping all database tables")
            Base.metadata.drop_all(bind=self.engine)
            logger.info("All tables dropped")

        except SQLAlchemyError as e:
            logger.error(f"Error dropping tables: {str(e)}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()

    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_session() as session:
                # Test basic connectivity
                result = session.execute(text("SELECT 1")).fetchone()

                # Get database info
                db_info = session.execute(text("SELECT version()")).fetchone()

                # Get pool status safely (different methods for SQLAlchemy 2.x)
                pool_status = {}
                try:
                    pool = self.engine.pool
                    # Try to get pool info if available
                    pool_status = {
                        "pool_size": getattr(pool, '_pool', None) and len(pool._pool.queue) if hasattr(pool, '_pool') else 0,
                        "status": "connected"
                    }
                except Exception:
                    pool_status = {"status": "unknown"}

                return {
                    "status": "healthy",
                    "connection_test": "passed",
                    "database_info": str(db_info[0]) if db_info else "unknown",
                    "pool_status": pool_status
                }

        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def execute_raw_sql(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute raw SQL query and return results."""
        try:
            with self.get_session() as session:
                result = session.execute(text(query), params or {})

                # Convert result to list of dictionaries
                if result.returns_rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in result.fetchall()]
                else:
                    return []

        except SQLAlchemyError as e:
            logger.error(f"Raw SQL execution failed: {str(e)}")
            raise

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table."""
        try:
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)

            with self.get_session() as session:
                # Get column information
                column_query = """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """
                columns = session.execute(text(column_query), {"table_name": validated_table_name}).fetchall()

                # Get row count - table name is validated, so safe to use in query
                count_query = text(f"SELECT COUNT(*) FROM {validated_table_name}")
                row_count = session.execute(count_query).scalar()

                return {
                    "table_name": validated_table_name,
                    "row_count": row_count,
                    "columns": [
                        {
                            "name": col[0],
                            "type": col[1],
                            "nullable": col[2] == "YES",
                            "default": col[3]
                        }
                        for col in columns
                    ]
                }

        except (SQLAlchemyError, SecurityError) as e:
            logger.error(f"Error getting table info for {table_name}: {str(e)}")
            raise DatabaseError(f"Failed to get table info for {table_name}", {"error": str(e)})
        except Exception as e:
            logger.error(f"Unexpected error getting table info for {table_name}: {str(e)}")
            raise DatabaseError(f"Unexpected error getting table info", {"error": str(e)})

    def backup_table_to_json(self, table_name: str, output_path: str):
        """Backup table data to JSON file."""
        import json
        from pathlib import Path

        try:
            # Validate table name to prevent SQL injection
            validated_table_name = self._validate_table_name(table_name)

            with self.get_session() as session:
                # Get all data from table - table name is validated, so safe to use
                query = text(f"SELECT * FROM {validated_table_name}")
                result = session.execute(query)
                columns = result.keys()
                data = [dict(zip(columns, row)) for row in result.fetchall()]

                # Write to JSON file
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

                logger.info(f"Table {validated_table_name} backed up to {output_path}")

        except (SecurityError, SQLAlchemyError) as e:
            logger.error(f"Backup failed for table {table_name}: {str(e)}")
            raise DatabaseError(f"Failed to backup table {table_name}", {"error": str(e)})
        except Exception as e:
            logger.error(f"Unexpected error during backup for table {table_name}: {str(e)}")
            raise DatabaseError(f"Unexpected error during backup", {"error": str(e)})

    def get_database_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database metrics."""
        try:
            with self.get_session() as session:
                metrics = {}

                # Get table sizes
                size_query = """
                    SELECT
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats
                    WHERE schemaname = 'public'
                """
                table_stats = session.execute(text(size_query)).fetchall()

                # Get database size
                db_size_query = "SELECT pg_size_pretty(pg_database_size(current_database()))"
                db_size = session.execute(text(db_size_query)).scalar()

                # Get connection count
                connection_query = """
                    SELECT count(*) as total_connections
                    FROM pg_stat_activity
                    WHERE state = 'active'
                """
                active_connections = session.execute(text(connection_query)).scalar()

                # Get table row counts for all our tables
                table_counts = {}
                for table_class in Base.__subclasses__():
                    table_name = table_class.__tablename__
                    try:
                        # Validate table name (should always be valid for our models, but safety first)
                        validated_table_name = self._validate_table_name(table_name)
                        count_query = text(f"SELECT COUNT(*) FROM {validated_table_name}")
                        count = session.execute(count_query).scalar()
                        table_counts[validated_table_name] = count
                    except SecurityError as e:
                        logger.warning(f"Skipping invalid table name {table_name}: {str(e)}")
                        continue

                return {
                    "database_size": db_size,
                    "active_connections": active_connections,
                    "table_counts": table_counts,
                    "table_statistics": [dict(zip(["schema", "table", "column", "n_distinct", "correlation"], row))
                                       for row in table_stats],
                    "pool_status": {
                        "pool_size": self.engine.pool.size(),
                        "checked_in": self.engine.pool.checkedin(),
                        "checked_out": self.engine.pool.checkedout(),
                    }
                }

        except Exception as e:
            logger.error(f"Error getting database metrics: {str(e)}")
            return {"error": str(e)}

    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")


# Database instance - will be initialized when config is available
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    global db_manager
    if db_manager is None:
        from ..utils.config import config
        db_manager = DatabaseManager(config)
    return db_manager


def get_db_session():
    """Dependency for getting database session in FastAPI."""
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session