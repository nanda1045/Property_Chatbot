-- One row per property. This is the root scope table used by the UI,
-- structured queries, and retrieval tools to keep every answer bounded to
-- the active property_code.
CREATE TABLE IF NOT EXISTS properties (
  property_code VARCHAR(32) PRIMARY KEY,
  property_name VARCHAR(255) NOT NULL,
  address VARCHAR(255) NULL,
  source_site VARCHAR(500) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- One row per imported rent-roll file/month. This table lets the assistant
-- identify the latest available snapshot, build month-over-month trends, and
-- reject requests for years or months that are not present in the data.
CREATE TABLE IF NOT EXISTS rent_roll_reports (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  property_code VARCHAR(32) NOT NULL,
  report_month DATE NOT NULL,
  as_of_date DATE NULL,
  source_file VARCHAR(1024) NOT NULL,
  source_file_hash CHAR(64) NOT NULL,
  source_filename VARCHAR(255) NOT NULL,
  imported_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_rent_roll_reports_source_hash (source_file_hash),
  KEY idx_rent_roll_reports_scope (property_code, report_month),
  CONSTRAINT fk_reports_property
    FOREIGN KEY (property_code) REFERENCES properties(property_code)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Unit-level rent-roll rows. This table powers detailed operational questions
-- such as vacant units, highest balances, average rent by unit type, square
-- footage, resident status, and lease timing.
CREATE TABLE IF NOT EXISTS rent_roll_units (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  report_id BIGINT NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  resident_group VARCHAR(128) NULL,
  unit VARCHAR(64) NOT NULL,
  unit_type VARCHAR(128) NULL,
  sqft INT NULL,
  resident_id VARCHAR(128) NULL,
  resident_name VARCHAR(255) NULL,
  resident_status VARCHAR(64) NULL,
  market_rent DECIMAL(12,2) NULL,
  resident_deposit DECIMAL(12,2) NULL,
  other_deposit DECIMAL(12,2) NULL,
  move_in_date DATE NULL,
  lease_expiration_date DATE NULL,
  move_out_date DATE NULL,
  balance DECIMAL(12,2) NULL,
  source_row_number INT NOT NULL,
  UNIQUE KEY uq_rent_roll_units_report_row (report_id, source_row_number),
  KEY idx_rent_roll_units_scope_unit (property_code, unit),
  KEY idx_rent_roll_units_scope_report (property_code, report_id),
  CONSTRAINT fk_units_report
    FOREIGN KEY (report_id) REFERENCES rent_roll_reports(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_units_property
    FOREIGN KEY (property_code) REFERENCES properties(property_code)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Unit-level lease charge rows tied back to a rent_roll_units record. This is
-- useful for charge drilldowns at the individual unit level and preserves the
-- detailed charge-code lines parsed from the rent-roll file.
CREATE TABLE IF NOT EXISTS lease_charges (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  rent_roll_unit_id BIGINT NOT NULL,
  report_id BIGINT NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  charge_code VARCHAR(64) NOT NULL,
  amount DECIMAL(12,2) NULL,
  source_row_number INT NOT NULL,
  KEY idx_lease_charges_scope_code (property_code, charge_code),
  KEY idx_lease_charges_report (report_id),
  CONSTRAINT fk_charges_unit
    FOREIGN KEY (rent_roll_unit_id) REFERENCES rent_roll_units(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_charges_report
    FOREIGN KEY (report_id) REFERENCES rent_roll_reports(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_charges_property
    FOREIGN KEY (property_code) REFERENCES properties(property_code)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Report-level summary KPI rows from the rent roll. This table powers high-level
-- answers such as occupancy, total units, market rent, lease charges, vacant unit
-- counts, balances, and occupancy trends over time.
CREATE TABLE IF NOT EXISTS rent_roll_summary_groups (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  report_id BIGINT NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  group_name VARCHAR(128) NOT NULL,
  square_footage DECIMAL(14,2) NULL,
  market_rent DECIMAL(14,2) NULL,
  lease_charges DECIMAL(14,2) NULL,
  security_deposit DECIMAL(14,2) NULL,
  other_deposits DECIMAL(14,2) NULL,
  unit_count INT NULL,
  unit_occupancy_pct DECIMAL(7,2) NULL,
  sqft_occupied_pct DECIMAL(7,2) NULL,
  balance DECIMAL(14,2) NULL,
  source_row_number INT NOT NULL,
  KEY idx_summary_groups_scope (property_code, group_name),
  CONSTRAINT fk_summary_groups_report
    FOREIGN KEY (report_id) REFERENCES rent_roll_reports(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_summary_groups_property
    FOREIGN KEY (property_code) REFERENCES properties(property_code)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Report-level charge totals grouped by charge code. This table powers charge
-- breakdown answers and charts, such as the biggest lease charge categories for
-- the latest month.
CREATE TABLE IF NOT EXISTS rent_roll_charge_summary (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  report_id BIGINT NOT NULL,
  property_code VARCHAR(32) NOT NULL,
  charge_code VARCHAR(64) NOT NULL,
  amount DECIMAL(14,2) NULL,
  source_row_number INT NOT NULL,
  KEY idx_charge_summary_scope_code (property_code, charge_code),
  CONSTRAINT fk_charge_summary_report
    FOREIGN KEY (report_id) REFERENCES rent_roll_reports(id)
    ON DELETE CASCADE,
  CONSTRAINT fk_charge_summary_property
    FOREIGN KEY (property_code) REFERENCES properties(property_code)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
