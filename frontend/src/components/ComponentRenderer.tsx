import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import type { UIComponent } from "../types";

type RecordValue = string | number | boolean | null | undefined;
type DataRecord = Record<string, RecordValue>;
type ChartRecord = {
  label: RecordValue;
  value: number;
  unit?: RecordValue;
  unit_count?: RecordValue;
};

function isRecord(value: unknown): value is DataRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isRecordArray(value: unknown): value is DataRecord[] {
  return Array.isArray(value) && value.every(isRecord);
}

function formatValue(value: RecordValue, unit?: RecordValue): string {
  if (typeof value !== "number") {
    return value == null ? "" : String(value);
  }

  if (unit === "USD") {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0
    }).format(value);
  }

  const formatted = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
  }).format(value);

  return unit ? `${formatted}${unit === "%" ? "%" : ` ${unit}`}` : formatted;
}

function formatCompactNumber(value: number, unit?: RecordValue): string {
  if (unit === "USD") {
    return new Intl.NumberFormat("en-US", {
      notation: "compact",
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 1
    }).format(value);
  }

  if (unit === "%") {
    return `${new Intl.NumberFormat("en-US", {
      maximumFractionDigits: 1
    }).format(value)}%`;
  }

  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

function formatRenderableValue(value: unknown, unit?: RecordValue): string {
  const numericValue = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "";
  }
  return formatValue(numericValue, unit);
}

function formatCompactRenderableValue(value: unknown, unit?: RecordValue): string {
  const numericValue = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(numericValue)) {
    return "";
  }
  return formatCompactNumber(numericValue, unit);
}

function formatColumnName(column: string): string {
  return column
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function inferUnit(title: string, data: ChartRecord[]): RecordValue {
  const explicitUnit = data.find((row) => row.unit)?.unit;
  if (explicitUnit) {
    return explicitUnit;
  }

  const normalizedTitle = title.toLowerCase();
  if (normalizedTitle.includes("rent") || normalizedTitle.includes("charge")) {
    return "USD";
  }
  if (normalizedTitle.includes("occupancy") || normalizedTitle.includes("percent")) {
    return "%";
  }
  return undefined;
}

function normalizeChartData(data: unknown): ChartRecord[] {
  if (!isRecordArray(data)) {
    return [];
  }

  return data
    .map((row) => ({
      label: row.label ?? row.report_month ?? row.unit_type ?? row.charge_code ?? "",
      value: typeof row.value === "number" ? row.value : Number(row.value ?? 0),
      unit: row.unit,
      unit_count: row.unit_count
    }))
    .filter((row) => row.label !== "" && Number.isFinite(row.value));
}

function chartDescription(title: string, unit?: RecordValue): string {
  const normalizedTitle = title.toLowerCase();
  if (normalizedTitle.includes("bedroom category")) {
    return "Each bar shows the average monthly market rent in USD by broad bedroom category.";
  }
  if (normalizedTitle.includes("floorplan code")) {
    return "Each bar shows the average monthly market rent in USD for a rent-roll floorplan code.";
  }
  if (normalizedTitle.includes("average market rent")) {
    return "Each bar shows the average monthly market rent in USD.";
  }
  if (normalizedTitle.includes("charge")) {
    return "Each bar shows the monthly charge amount in USD for that category.";
  }
  if (normalizedTitle.includes("occupancy")) {
    return "Each point shows unit occupancy percentage for that report month.";
  }
  if (unit === "USD") {
    return "Values are shown in USD.";
  }
  if (unit === "%") {
    return "Values are shown as percentages.";
  }
  return "";
}

function chartLabelName(title: string): string {
  const normalizedTitle = title.toLowerCase();
  if (normalizedTitle.includes("bedroom category")) {
    return "Bedroom category";
  }
  if (normalizedTitle.includes("floorplan code")) {
    return "Floorplan code";
  }
  if (normalizedTitle.includes("charge")) {
    return "Charge category";
  }
  if (normalizedTitle.includes("occupancy")) {
    return "Report month";
  }
  return "Label";
}

function formatTableValue(column: string, value: RecordValue): string {
  if (typeof value !== "number") {
    return value == null ? "" : String(value);
  }

  const normalizedColumn = column.toLowerCase();
  if (
    normalizedColumn.includes("balance") ||
    normalizedColumn.includes("rent") ||
    normalizedColumn.includes("amount") ||
    normalizedColumn.includes("charge") ||
    normalizedColumn.includes("deposit")
  ) {
    return formatValue(value, "USD");
  }

  if (normalizedColumn.includes("pct") || normalizedColumn.includes("percent")) {
    return formatValue(value, "%");
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: Number.isInteger(value) ? 0 : 2
  }).format(value);
}

function orderColumns(columns: string[]): string[] {
  const preferred = [
    "unit",
    "balance",
    "market_rent",
    "resident_status",
    "sqft",
    "bedroom_category",
    "unit_type",
    "report_month"
  ];
  return [
    ...preferred.filter((column) => columns.includes(column)),
    ...columns.filter((column) => !preferred.includes(column))
  ];
}

function KpiCard({ component }: { component: UIComponent }) {
  const data = isRecord(component.data) ? component.data : {};
  return (
    <section className="component-panel kpi-panel">
      <span className="component-title">{component.title}</span>
      <strong>{formatValue(data.value, data.unit)}</strong>
      {data.report_month ? <small>{String(data.report_month)}</small> : null}
    </section>
  );
}

function ChartPanel({ component, variant }: { component: UIComponent; variant: "bar" | "line" }) {
  const data = normalizeChartData(component.data);
  const unit = inferUnit(component.title, data);
  const useVerticalBars = variant === "bar" && data.length > 6;
  const chartHeight = useVerticalBars ? Math.max(300, data.length * 34 + 72) : 280;
  const description = component.description || chartDescription(component.title, unit);
  const labelName = chartLabelName(component.title);

  if (data.length === 0) {
    return <JsonPanel component={component} />;
  }

  return (
    <section className="component-panel chart-panel">
      <div className="component-heading">
        <h3>{component.title}</h3>
        {description ? <p>{description}</p> : null}
      </div>
      <div className="chart-frame" style={{ height: chartHeight }}>
        <ResponsiveContainer width="100%" height="100%">
          {variant === "bar" && useVerticalBars ? (
            <BarChart data={data} layout="vertical" margin={{ left: 20, right: 92 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                type="number"
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => formatCompactNumber(Number(value), unit)}
              />
              <YAxis
                type="category"
                dataKey="label"
                tickLine={false}
                axisLine={false}
                width={118}
              />
              <Tooltip
                formatter={(value) => [formatValue(Number(value), unit), "Value"]}
                labelFormatter={(label) => `${labelName}: ${label}`}
              />
              <Bar dataKey="value" fill="#2f7d64" radius={[0, 4, 4, 0]}>
                <LabelList
                  dataKey="value"
                  position="right"
                  formatter={(value) => formatRenderableValue(value, unit)}
                  className="bar-value-label"
                />
              </Bar>
            </BarChart>
          ) : variant === "bar" ? (
            <BarChart data={data} margin={{ right: 18 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tickLine={false} axisLine={false} interval={0} />
              <YAxis
                tickLine={false}
                axisLine={false}
                width={72}
                tickFormatter={(value) => formatCompactNumber(Number(value), unit)}
              />
              <Tooltip formatter={(value) => [formatValue(Number(value), unit), "Value"]} />
              <Bar dataKey="value" fill="#2f7d64" radius={[4, 4, 0, 0]}>
                <LabelList
                  dataKey="value"
                  position="top"
                  formatter={(value) => formatCompactRenderableValue(value, unit)}
                  className="bar-value-label"
                />
              </Bar>
            </BarChart>
          ) : (
            <LineChart data={data} margin={{ right: 18 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={24} />
              <YAxis
                tickLine={false}
                axisLine={false}
                width={64}
                tickFormatter={(value) => formatCompactNumber(Number(value), unit)}
              />
              <Tooltip formatter={(value) => formatValue(Number(value), unit)} />
              <Line type="monotone" dataKey="value" stroke="#315f9c" strokeWidth={2} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </section>
  );
}

function TablePanel({ component }: { component: UIComponent }) {
  const rows = isRecordArray(component.data) ? component.data : [];
  const columns = rows[0] ? orderColumns(Object.keys(rows[0])) : [];

  if (rows.length === 0 || columns.length === 0) {
    return <JsonPanel component={component} />;
  }

  return (
    <section className="component-panel table-panel">
      <h3>{component.title}</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{formatColumnName(column)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${component.title}-${rowIndex}`}>
                {columns.map((column) => (
                  <td key={column}>{formatTableValue(column, row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ComparisonPanel({ component }: { component: UIComponent }) {
  const rows = isRecordArray(component.data) ? component.data : [];

  if (rows.length === 0) {
    return <JsonPanel component={component} />;
  }

  return (
    <section className="component-panel comparison-panel">
      <h3>{component.title}</h3>
      <div className="comparison-grid">
        {rows.map((row, index) => (
          <div className="comparison-item" key={`${component.title}-${index}`}>
            {Object.entries(row)
              .filter(([key]) => key !== "unit")
              .map(([key, value]) => (
                <p key={key}>
                  <span>{key.replaceAll("_", " ")}</span>
                  <strong>{formatValue(value, key === "value" ? row.unit : undefined)}</strong>
                </p>
              ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function SqlApprovalPanel({
  component,
  onApprove
}: {
  component: UIComponent;
  onApprove?: (component: UIComponent) => void;
}) {
  const data = isRecord(component.data) ? component.data : {};
  const sql = typeof data.sql === "string" ? data.sql : "";
  const explanation = typeof data.explanation === "string" ? data.explanation : component.description;
  const status = typeof data.status === "string" ? data.status : "pending_approval";
  const executable = data.executable !== false;

  return (
    <section className="component-panel sql-approval-panel">
      <div className="component-heading">
        <h3>{component.title}</h3>
        {explanation ? <p>{explanation}</p> : null}
      </div>
      <p>
        <strong>Status:</strong> {formatColumnName(status)}
      </p>
      <pre>{sql}</pre>
      {executable ? (
        <button type="button" onClick={() => onApprove?.(component)} disabled={!sql || !onApprove}>
          Run approved query
        </button>
      ) : null}
    </section>
  );
}

function JsonPanel({ component }: { component: UIComponent }) {
  return (
    <section className="component-panel json-panel">
      <h3>{component.title}</h3>
      <pre>{JSON.stringify(component.data, null, 2)}</pre>
    </section>
  );
}

export function ComponentRenderer({
  component,
  onApprove
}: {
  component: UIComponent;
  onApprove?: (component: UIComponent) => void;
}) {
  switch (component.type) {
    case "kpi_card":
      return <KpiCard component={component} />;
    case "bar_chart":
      return <ChartPanel component={component} variant="bar" />;
    case "line_chart":
      return <ChartPanel component={component} variant="line" />;
    case "table":
      return <TablePanel component={component} />;
    case "comparison_view":
      return <ComparisonPanel component={component} />;
    case "sql_approval":
      return <SqlApprovalPanel component={component} onApprove={onApprove} />;
    default:
      return <JsonPanel component={component} />;
  }
}
