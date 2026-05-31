-- properties
-- What it stores: one master record per property, keyed by property_code.
-- Why it is useful: this is the root scoping table for the whole assistant.
-- The UI, SQL tools, retrieval filters, and response metadata all use the
-- selected property_code from this table so answers stay property-specific.
CREATE TABLE IF NOT EXISTS properties (
  property_code VARCHAR(32) PRIMARY KEY,
  property_name VARCHAR(255) NOT NULL,
  address VARCHAR(255) NULL,
  source_site VARCHAR(500) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- rent_roll_reports
-- What it stores: one record for each imported rent-roll file/month for a
-- property, including source file metadata and report_month.
-- Why it is useful: this table defines the available reporting snapshots. It
-- lets the assistant find the latest month, build trends across months, audit
-- where data came from, and avoid answering for periods that were not loaded.
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

-- rent_roll_units
-- What it stores: one row per unit in each rent-roll report, including unit
-- number, unit type, square footage, status, market rent, balance, and lease
-- dates.
-- Why it is useful: this is the main unit-level analytics table. It powers
-- questions about vacant units, highest balances, rent by unit type, unit mix,
-- square footage, occupancy status, and other operational details.
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

-- lease_charges
-- What it stores: detailed lease charge lines for each unit/report, tied back
-- to rent_roll_units by rent_roll_unit_id.
-- Why it is useful: this preserves the granular charge-code data from the
-- source rent roll. It can support future drilldowns like charges by unit,
-- charge history for a unit, or charge-category analysis at a detailed level.
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

-- rent_roll_summary_groups
-- What it stores: report-level summary rows from the rent roll, usually grouped
-- by summary categories such as total property, occupied, vacant, or other
-- rent-roll groupings.
-- Why it is useful: this table powers high-level KPI answers such as occupancy,
-- total units, vacant unit count, market rent, lease charges, balances, and
-- month-over-month occupancy trends without recalculating everything from raw
-- unit rows each time.
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

-- rent_roll_charge_summary
-- What it stores: report-level charge totals grouped by charge_code for each
-- property/month.
-- Why it is useful: this table powers charge breakdown answers and charts, such
-- as top lease charge categories, rent vs fee composition, and latest-month
-- charge totals. It is faster and cleaner than summing every unit-level charge
-- line for common dashboard questions.
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
