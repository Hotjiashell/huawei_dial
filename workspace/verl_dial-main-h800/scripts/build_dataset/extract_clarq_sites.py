import json
from collections import Counter
from html import unescape
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from tqdm import tqdm


INPUT_PATH = Path("ClarQ/train.json")
OUTPUT_PATH = Path("ClarQ/train_sites.json")
SITES_API = "https://api.stackexchange.com/2.3/sites"


def count_site_prefixes() -> Counter[str]:
    counts: Counter[str] = Counter()

    with INPUT_PATH.open("rb") as file, tqdm(
        total=INPUT_PATH.stat().st_size,
        desc="Reading ClarQ",
        unit="B",
        unit_scale=True,
    ) as progress:
        for line in file:
            sample_id = json.loads(line)["id"]
            counts[sample_id.rsplit("_", 2)[0]] += 1
            progress.update(len(line))

    return counts


def fetch_stack_exchange_sites() -> list[dict]:
    sites = []
    page = 1

    with tqdm(desc="Fetching site metadata", unit="page") as progress:
        while True:
            query = urlencode({"page": page, "pagesize": 100, "filter": "default"})
            with urlopen(f"{SITES_API}?{query}") as response:
                payload = json.load(response)

            sites.extend(payload["items"])
            progress.update()
            if not payload["has_more"]:
                return sites

            page += 1


def get_site_prefixes(site: dict) -> set[str]:
    prefixes = {site["api_site_parameter"]}

    for url in [site["site_url"], *site.get("aliases", [])]:
        hostname = urlparse(url).hostname
        prefixes.add(hostname.split(".")[0])

    if site["site_type"] == "main_site":
        prefixes.add(site["api_site_parameter"].split(".")[0])

    return prefixes


def match_site_metadata(
    site_counts: Counter[str],
    stack_exchange_sites: list[dict],
) -> dict[str, dict]:
    metadata = {}

    for site in stack_exchange_sites:
        for prefix in get_site_prefixes(site) & site_counts.keys():
            metadata[prefix] = site

    try:
        missing_prefixes = site_counts.keys() - metadata.keys()
        if missing_prefixes:
            raise KeyError(
                f"Prefixes not found in the Stack Exchange Sites API: "
                f"{sorted(missing_prefixes)}"
            )
    except KeyError as error:
        print(error.args[0])

    return metadata


def main() -> None:
    site_counts = count_site_prefixes()
    site_metadata = match_site_metadata(
        site_counts,
        fetch_stack_exchange_sites(),
    )

    sites = []
    for prefix, count in site_counts.most_common():
        metadata = site_metadata.get(prefix, {})
        sites.append(
            {
                "site_prefix": prefix,
                "site_name": unescape(metadata.get("name", "")),
                "site_description": unescape(metadata.get("audience", "")),
                "entry_count": count,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(sites, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
