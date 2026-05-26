# Unstructured Data Source Plan

The structured rent-roll files contain property names and property codes. For unstructured retrieval, use public property-specific leasing websites, not broad web search during user questions. The scraper should only crawl the `primary_site` and `seed_paths` configured for the active property code in `config/property_sources.json`.

## Recommended Sources

| Codes | Property | Public source |
| --- | --- | --- |
| `115r` | Canfield Park | https://canfield-park.com/ |
| `126a`, `126r` | The Halden | https://www.thehalden.com/ |
| `134c`, `134land`, `134r` | 55 Riverwalk Place | https://55riverwalkplace.com/ |
| `138a`, `138r` | Everbend | https://everbendny.com/ |
| `139c`, `139r` | The Mill Greenwich | https://themillgreenwich.com/ |
| `143a`, `143c` | The Ellsworth | https://ellsworthny.com/ |
| `144r` | Winner's Circle at Saratoga | https://winnerscircleatsaratoga.com/ |
| `153a`, `153c`, `153r` | Abbot Mill | https://abbotmill.com/ |
| `175r` | Kinwood | https://kinwoodny.com/ |
| `176r` | The Alexander | https://alexanderalbany.com/ |
| `183a`, `183c`, `183r` | Luckey Platt | https://luckeyplatt.com/ |
| `184r` | Lakeshore Preserve | https://livelakeshorepreserve.com/ |
| `185r` | Waterfront at the Strand | https://livewaterfrontstrand.com/ |
| `462a` | Stony Run at the Stockade | https://stonyrunstockade.com/ |
| `altapm` | AltaPM | No property-specific public source found yet |

## What To Scrape

For a representative sample, crawl only a handful of pages per property:

- Home page for positioning, overview, phone, and address.
- Floor plans page for bedroom mix, square footage ranges, pricing language, and availability copy.
- Amenities page for community and apartment features.
- Gallery page for image alt text and captions if available.
- Neighborhood page for local context.
- Contact page for canonical address, office hours, and phone number.

## Scoping Rule

Every scraped document chunk should carry:

- `property_code`
- `property_name`
- `source_url`
- `page_type`
- `section_heading`
- `chunk_strategy`
- `scraped_at`

At query time, retrieval must filter by `property_code = active_property_code` before similarity ranking. If several codes map to the same marketing site, keep the exact active code on the chunk or duplicate the small document set per code. This keeps the demo aligned with the requirement that every retrieval call is bounded to the active property code.

## Fallback Sources

If an official site is temporarily unavailable, use public listing pages only as fallback, such as Apartments.com, Zillow, or RentCafe. These should be ingested with the same `property_code` filter and marked with `source_type = listing_site` so the assistant can prefer official-site content when both are available.
