# update_master_rss.py

import requests
import json
import os
import yaml
import sys
import re


def load_config(path='feeds_config.yaml'):
    """
    Load configuration from a YAML file.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_master(master_path):
    """
    Load existing master RSS list from JSON, or return empty list.
    """
    if os.path.exists(master_path):
        with open(master_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_master(data, master_path):
    """
    Save the master RSS list to JSON.
    """
    with open(master_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fetch_top_charts(country, limit):
    """
    Fetch top podcasts from Apple Marketing Tools RSS endpoint.
    Returns list of dicts with 'name' and 'url'.
    """
    url = f'https://rss.applemarketingtools.com/api/v2/{country}/podcasts/top/{limit}/podcasts.json'
    resp = requests.get(url)
    resp.raise_for_status()
    items = resp.json().get('feed', {}).get('results', [])
    return [{'name': item['name'], 'url': item['url']} for item in items]


def fetch_search_podcasts(term, limit):
    """
    Use the iTunes Search API to fetch podcasts matching a search term.
    Returns list of dicts with 'name' and 'url'.
    """
    resp = requests.get(
        'https://itunes.apple.com/search',
        params={'media': 'podcast', 'term': term, 'limit': limit}
    )
    resp.raise_for_status()
    results = resp.json().get('results', [])
    return [
        {'name': r.get('collectionName'), 'url': r.get('feedUrl')}
        for r in results if r.get('feedUrl')
    ]


def resolve_apple_feed(url):
    """
    If the URL is an Apple Podcasts page, use the iTunes Lookup API to get the real RSS feed URL.
    Otherwise return the original URL.
    """
    if 'podcasts.apple.com' in url:
        m = re.search(r'/id(\d+)', url)
        if m:
            pid = m.group(1)
            try:
                lookup = requests.get(
                    'https://itunes.apple.com/lookup',
                    params={'id': pid},
                    timeout=10
                )
                lookup.raise_for_status()
                results = lookup.json().get('results', [])
                if results and results[0].get('feedUrl'):
                    return results[0]['feedUrl']
            except Exception as e:
                print(f"iTunes lookup failed for {url}: {e}", file=sys.stderr)
    return url


def update_master(master, new_entries):
    """
    Merge new entries into the master list, avoiding duplicates by URL.
    """
    existing = {e['url'] for e in master}
    for entry in new_entries:
        orig_url = entry.get('url')
        if not orig_url:
            continue
        # Resolve Apple page links
        real_url = resolve_apple_feed(orig_url)
        entry['url'] = real_url
        if real_url in existing:
            continue
        master.append(entry)
        existing.add(real_url)
    return master


def main():
    cfg = load_config()
    feeds_cfg    = cfg['feeds']
    master_file  = feeds_cfg.get('master_file', 'master_rss.json')
    country      = feeds_cfg.get('country', 'us')
    top_limit    = feeds_cfg.get('top_chart_limit', 10)
    search_terms = feeds_cfg.get('search_terms', [])
    search_limit = feeds_cfg.get('search_limit', 10)

    master = load_master(master_file)
    new    = fetch_top_charts(country, top_limit)
    for term in search_terms:
        new += fetch_search_podcasts(term, search_limit)

    master = update_master(master, new)
    save_master(master, master_file)
    print(f"Master list updated. {len(master)} feeds saved to {master_file}")

if __name__ == '__main__':
    main()