# Structured Rent-Roll MySQL Load

The structured rent-roll spreadsheets are loaded into MySQL with every table keyed or indexed by `property_code` so chatbot tools can enforce property scope.

## Start MySQL

```bash
docker compose up -d mysql
```

The local Compose service uses:

- host: `127.0.0.1`
- port: `3306`
- database: `aker_chatbot`
- user: `root`
- password: `root`

## Load Data

```bash
python3 scripts/load_rent_roll_mysql.py --reset
```

The loader parses the `.xls` files directly as zipped Office XML workbooks, so it does not require `pandas`, `openpyxl`, or a Python MySQL driver. It uses the local `mysql` CLI for the final import.

To generate a SQL load file without applying it:

```bash
SKIP_MYSQL_LOAD=1 python3 scripts/load_rent_roll_mysql.py --reset --write-sql data/structured/rent_roll_load.sql
```

## Loaded Tables

- `properties`
- `rent_roll_reports`
- `rent_roll_units`
- `lease_charges`
- `rent_roll_summary_groups`
- `rent_roll_charge_summary`

`property_code` is present on every table that the chatbot will query.

## Verification Query

```bash
MYSQL_PWD=root mysql --host=127.0.0.1 --port=3306 --user=root --protocol=tcp aker_chatbot \
  -e "SELECT 'properties' AS table_name, COUNT(*) AS row_count FROM properties
      UNION ALL SELECT 'rent_roll_reports', COUNT(*) FROM rent_roll_reports
      UNION ALL SELECT 'rent_roll_units', COUNT(*) FROM rent_roll_units
      UNION ALL SELECT 'lease_charges', COUNT(*) FROM lease_charges
      UNION ALL SELECT 'rent_roll_summary_groups', COUNT(*) FROM rent_roll_summary_groups
      UNION ALL SELECT 'rent_roll_charge_summary', COUNT(*) FROM rent_roll_charge_summary;"
```

Expected load counts from the provided data:

| Table | Rows |
| --- | ---: |
| `properties` | 25 |
| `rent_roll_reports` | 300 |
| `rent_roll_units` | 43,058 |
| `lease_charges` | 94,491 |
| `rent_roll_summary_groups` | 1,800 |
| `rent_roll_charge_summary` | 1,156 |
