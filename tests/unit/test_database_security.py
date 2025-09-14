"""
Unit tests for database security features.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from database.database import DatabaseManager
from utils.exceptions import SecurityError, DatabaseError


class TestDatabaseSecurity:
    """Test database security features."""

    @pytest.fixture
    def mock_db_manager(self, test_config):
        """Create a mock database manager for testing."""
        with patch('database.database.create_engine') as mock_engine:
            mock_engine.return_value = Mock()
            db_manager = DatabaseManager(test_config)
            db_manager.engine = Mock()
            return db_manager

    def test_validate_table_name_valid_names(self, mock_db_manager):
        """Test validation of valid table names."""
        # Set up valid table names
        mock_db_manager._valid_table_names = {
            "users", "experiment_runs", "model_deployments"
        }

        # Test valid table names
        assert mock_db_manager._validate_table_name("users") == "users"
        assert mock_db_manager._validate_table_name("experiment_runs") == "experiment_runs"
        assert mock_db_manager._validate_table_name("model_deployments") == "model_deployments"

    def test_validate_table_name_invalid_characters(self, mock_db_manager):
        """Test validation rejects table names with invalid characters."""
        mock_db_manager._valid_table_names = {"users"}

        # Test table names with invalid characters
        invalid_names = [
            "users; DROP TABLE users;",  # SQL injection attempt
            "users--",  # SQL comment
            "users/*",  # SQL comment
            "users'",  # Single quote
            'users"',  # Double quote
            "users\n",  # Newline
            "users\t",  # Tab
            "users ",  # Space
            "123users",  # Starting with number
            "",  # Empty string
        ]

        for invalid_name in invalid_names:
            with pytest.raises(SecurityError, match="Invalid table name format"):
                mock_db_manager._validate_table_name(invalid_name)

    def test_validate_table_name_not_in_whitelist(self, mock_db_manager):
        """Test validation rejects table names not in whitelist."""
        mock_db_manager._valid_table_names = {"users", "products"}

        # Test table name not in whitelist
        with pytest.raises(SecurityError, match="Table name not allowed"):
            mock_db_manager._validate_table_name("malicious_table")

        # Verify error contains helpful information
        try:
            mock_db_manager._validate_table_name("malicious_table")
        except SecurityError as e:
            assert "valid_tables" in e.details
            assert e.details["valid_tables"] == ["users", "products"]

    def test_get_table_info_security(self, mock_db_manager):
        """Test get_table_info method uses validation."""
        mock_db_manager._valid_table_names = {"users"}

        # Mock session and query results
        mock_session = Mock()
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            ("id", "integer", "NO", None),
            ("name", "varchar", "YES", None)
        ]
        mock_session.execute.side_effect = [mock_result, Mock(scalar=lambda: 100)]

        with patch.object(mock_db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session

            # Test with valid table name
            result = mock_db_manager.get_table_info("users")
            assert result["table_name"] == "users"
            assert result["row_count"] == 100

            # Test with invalid table name
            with pytest.raises(DatabaseError):
                mock_db_manager.get_table_info("malicious; DROP TABLE users;")

    def test_backup_table_security(self, mock_db_manager, temp_dir):
        """Test backup_table_to_json method uses validation."""
        mock_db_manager._valid_table_names = {"users"}

        # Mock session and query results
        mock_session = Mock()
        mock_result = Mock()
        mock_result.keys.return_value = ["id", "name"]
        mock_result.fetchall.return_value = [(1, "Alice"), (2, "Bob")]
        mock_session.execute.return_value = mock_result

        with patch.object(mock_db_manager, 'get_session') as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = mock_session

            # Test with valid table name
            output_path = str(temp_dir / "backup.json")
            mock_db_manager.backup_table_to_json("users", output_path)

            # Test with invalid table name
            with pytest.raises(DatabaseError):
                mock_db_manager.backup_table_to_json("malicious'; DROP TABLE users;--", output_path)

    def test_database_metrics_security(self, mock_db_manager):
        """Test get_database_metrics method handles table validation."""
        # Set up mock for our model classes
        mock_table_class1 = Mock()
        mock_table_class1.__tablename__ = "users"
        mock_table_class2 = Mock()
        mock_table_class2.__tablename__ = "products"

        mock_db_manager._valid_table_names = {"users", "products"}

        with patch('database.database.Base.__subclasses__', return_value=[mock_table_class1, mock_table_class2]):
            mock_session = Mock()

            # Mock the various queries
            mock_session.execute.side_effect = [
                Mock(fetchall=lambda: []),  # table_stats query
                Mock(scalar=lambda: "1GB"),  # db_size query
                Mock(scalar=lambda: 5),  # active_connections query
                Mock(scalar=lambda: 100),  # count for users table
                Mock(scalar=lambda: 50),   # count for products table
            ]

            with patch.object(mock_db_manager, 'get_session') as mock_get_session:
                mock_get_session.return_value.__enter__.return_value = mock_session

                result = mock_db_manager.get_database_metrics()

                assert "table_counts" in result
                assert result["table_counts"]["users"] == 100
                assert result["table_counts"]["products"] == 50

    def test_table_name_validation_initialization(self, test_config):
        """Test that table name validation is properly initialized."""
        with patch('database.database.create_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            # Mock metadata reflection
            with patch('database.database.MetaData') as mock_metadata_class:
                mock_metadata = Mock()
                mock_metadata_class.return_value = mock_metadata
                mock_metadata.tables.keys.return_value = ["existing_table1", "existing_table2"]

                db_manager = DatabaseManager(test_config)

                # Trigger validation initialization by calling _validate_table_name
                db_manager.engine = mock_engine  # Set engine for validation
                with patch('database.database.Base.__subclasses__', return_value=[]):
                    try:
                        db_manager._validate_table_name("existing_table1")
                    except SecurityError:
                        pass  # Expected since table won't be in model tables

                # Verify metadata was reflected
                mock_metadata.reflect.assert_called_once_with(bind=mock_engine)

    def test_error_handling_in_validation(self, mock_db_manager):
        """Test error handling in database operations with validation."""
        mock_db_manager._valid_table_names = {"users"}

        # Test that SecurityError is properly converted to DatabaseError
        with patch.object(mock_db_manager, '_validate_table_name', side_effect=SecurityError("Invalid table")):
            with pytest.raises(DatabaseError):
                mock_db_manager.get_table_info("malicious_table")

        # Test that SQLAlchemyError is properly handled
        with patch.object(mock_db_manager, '_validate_table_name', return_value="users"):
            with patch.object(mock_db_manager, 'get_session', side_effect=Exception("DB Error")):
                with pytest.raises(DatabaseError):
                    mock_db_manager.get_table_info("users")