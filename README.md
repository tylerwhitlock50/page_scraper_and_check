# Page Keyword Auditor

This repository provides a small Playwright-based crawler that reviews every
page on a site for references to bicycle wheel terminology. It is designed to
help track down stray mentions of another business (for example, accidental
references to a different wheel brand) after outsourcing website development.

## Prerequisites

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the Playwright browser binaries (only required once per machine):

   ```bash
   playwright install chromium
   ```

## Usage

Run the auditor by passing the base site URL. By default it scans for a list of
wheel-related keywords, but you can supply your own with repeated `--keyword`
arguments.

```bash
python site_keyword_audit.py https://fandbsports.com \
  --keyword wheel --keyword rim --keyword bike
```

Useful optional flags:

- `--max-pages` limits the crawl depth if you only want to sample the site.
- `--delay` inserts a small pause between requests to avoid stressing the site.
- `--output` writes a JSON report of every match that was found.

Example output:

```
URL: https://fandbsports.com/
  - [title] 'wheel': Wheelsets for every rider
  - [body] 'bike': ...
```

If no matches are found the script prints a simple confirmation message. When an
output path is provided, the JSON array contains one entry per page with each
keyword match (including where it was found and a short snippet of context).
