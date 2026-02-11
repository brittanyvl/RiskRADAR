"""
Deep analysis of aircraft extraction gaps.

Goals:
1. Find ALL reports missing aircraft (with or without taxonomy)
2. Analyze titles to identify missed patterns
3. Find model number gaps (have make but no model)
4. Identify dirty text patterns (DC8 vs DC-8)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import sqlite3
import re
from collections import Counter

conn = sqlite3.connect('sqlite/riskradar.db')
cursor = conn.cursor()

print("=" * 70)
print("DEEP GAP ANALYSIS - Aircraft Extraction")
print("=" * 70)

# 1. ALL reports without aircraft
print("\n1. REPORTS WITHOUT AIRCRAFT CATEGORY")
print("-" * 50)

no_aircraft = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_make, f.aircraft_model
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
    WHERE f.aircraft_category IS NULL OR f.aircraft_category = ''
    ORDER BY f.report_id
""").fetchall()

print(f"   Total without aircraft: {len(no_aircraft)}")
print("\n   Titles:")
for rid, title, make, model in no_aircraft:
    print(f"   [{rid}] {title[:80]}...")

# 2. Reports WITH make but missing/incomplete model
print("\n\n2. REPORTS WITH MAKE BUT MISSING/INCOMPLETE MODEL")
print("-" * 50)

incomplete_model = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_make, f.aircraft_model, f.aircraft_category
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
    WHERE f.aircraft_make IS NOT NULL
      AND (f.aircraft_model IS NULL OR f.aircraft_model = '' OR LENGTH(f.aircraft_model) < 3)
    ORDER BY f.aircraft_make, f.report_id
""").fetchall()

print(f"   Total with incomplete model: {len(incomplete_model)}")

# Group by make
by_make = {}
for rid, title, make, model, cat in incomplete_model:
    if make not in by_make:
        by_make[make] = []
    by_make[make].append((rid, title, model))

print("\n   By manufacturer:")
for make in sorted(by_make.keys(), key=lambda x: -len(by_make[x])):
    print(f"\n   {make} ({len(by_make[make])} reports):")
    for rid, title, model in by_make[make][:5]:
        print(f"      [{rid}] model='{model}' | {title[:60]}...")
    if len(by_make[make]) > 5:
        print(f"      ... and {len(by_make[make]) - 5} more")

# 3. Pattern analysis - what aircraft text is in titles we're missing?
print("\n\n3. AIRCRAFT PATTERNS IN MISSED TITLES")
print("-" * 50)

# Common aircraft patterns to search for
aircraft_patterns = [
    # Cessna variants
    (r"Cessna[- ]?\d{3}", "Cessna with model"),
    (r"C[- ]?\d{3}[A-Z]?(?!\d)", "C-### pattern"),

    # Learjet/Lear Jet variants
    (r"Lear\s*[Jj]et[- ]?\d+", "Learjet with model"),
    (r"Lear\s*[Jj]et\s+Model\s+\d+", "Learjet Model ##"),
    (r"LR-?\d+[A-Z]?", "LR## pattern"),

    # Boeing variants
    (r"Boeing\s+\d{3}", "Boeing ###"),
    (r"B-?\d{3}", "B-### pattern"),
    (r"7[0-9]{2}", "7## pattern"),

    # Douglas/McDonnell Douglas
    (r"DC-?\d+", "DC-# pattern"),
    (r"MD-?\d+", "MD-## pattern"),
    (r"McDonnell\s+Douglas", "McDonnell Douglas"),

    # De Havilland
    (r"DHC-?\d+", "DHC-# pattern"),
    (r"DH-?\d+", "DH-### pattern"),
    (r"[Dd]e\s*[Hh]avilland", "de Havilland"),

    # Beech/Beechcraft
    (r"Beech(craft)?[- ]?\d+", "Beech with model"),
    (r"King\s+Air[- ]?\d+", "King Air with model"),
    (r"Baron[- ]?\d+", "Baron with model"),
    (r"B-?\d{2}[A-Z]?(?!\d)", "B## pattern (Beech)"),

    # Piper
    (r"PA-?\d+-?\d*", "PA-## pattern"),
    (r"Piper\s+\w+", "Piper with name"),

    # Gulfstream
    (r"Gulfstream\s+[IVXG]+", "Gulfstream with model"),
    (r"G-?[IVX]+\d*", "G-IV pattern"),
    (r"G\d{3}", "G### pattern"),

    # Embraer
    (r"EMB-?\d+", "EMB-### pattern"),
    (r"ERJ-?\d+", "ERJ-### pattern"),
    (r"Embraer\s+\w+", "Embraer with model"),

    # Fairchild
    (r"FH-?\d+", "FH-### pattern"),
    (r"F-?\d+[A-Z]?(?=\s|,|$)", "F-## pattern"),
    (r"Metro\s*[IVX]*", "Metro pattern"),

    # ATR
    (r"ATR[- ]?\d+", "ATR-## pattern"),

    # CASA
    (r"CASA[- ]?C?-?\d+", "CASA pattern"),
    (r"C-?\d{3}[A-Z]*", "C-### pattern"),

    # British Aerospace / BAe
    (r"BAe?[- ]?\d+", "BAe-### pattern"),
    (r"J-?\d{4}", "J-#### pattern (Jetstream)"),
    (r"Jetstream\s*\d*", "Jetstream pattern"),

    # Lockheed
    (r"L-?\d{3,4}", "L-#### pattern"),
    (r"Lockheed\s+\w+", "Lockheed with model"),

    # Sikorsky
    (r"S-?\d{2}[A-Z]?", "S-## pattern"),
    (r"Sikorsky", "Sikorsky"),

    # Bell
    (r"Bell\s+\d+", "Bell with model"),
    (r"UH-?\d+", "UH-# pattern"),

    # Eurocopter/Airbus Helicopters
    (r"EC-?\d+", "EC-### pattern"),
    (r"AS-?\d+", "AS-### pattern"),
    (r"Eurocopter", "Eurocopter"),

    # Convair
    (r"Convair\s+\d+", "Convair with model"),
    (r"CV-?\d+", "CV-### pattern"),

    # Grumman
    (r"G-?\d{2,3}[A-Z]?", "G-## pattern (Grumman)"),
    (r"Grumman\s+\w+", "Grumman with model"),

    # Mitsubishi
    (r"MU-?\d+[A-Z]?", "MU-## pattern"),

    # Swearingen
    (r"SA-?\d+", "SA-### pattern"),
    (r"Metro", "Metro"),

    # Nord
    (r"Nord\s+\d+", "Nord with model"),

    # Martin
    (r"Martin\s+\d+", "Martin with model"),

    # SAAB
    (r"SAAB\s+\d+", "SAAB with model"),

    # Fokker
    (r"Fokker\s+F?-?\d+", "Fokker pattern"),
    (r"F-?\d{2}(?!\d)", "F-## pattern (Fokker)"),

    # Cirrus
    (r"SR-?\d+", "SR-## pattern"),
    (r"Cirrus", "Cirrus"),

    # Rolls Royce (engines but sometimes in titles)
    (r"Rolls[- ]?Royce", "Rolls-Royce"),

    # Generic number patterns
    (r"Model\s+\d+", "Model ## pattern"),
    (r"N\d+[A-Z]+", "N-number"),
]

# Check which patterns match in the no_aircraft titles
print("\n   Patterns found in titles WITHOUT aircraft extraction:")
pattern_matches = Counter()
for rid, title, make, model in no_aircraft:
    if not title:
        continue
    for pattern, name in aircraft_patterns:
        if re.search(pattern, title, re.IGNORECASE):
            pattern_matches[name] += 1

for name, count in pattern_matches.most_common(20):
    print(f"      {name}: {count}")

# 4. Detailed look at specific problem areas
print("\n\n4. DETAILED PROBLEM AREAS")
print("-" * 50)

# Learjet issues
print("\n   A) LEARJET MODEL ISSUES:")
learjet_issues = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_model
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
    WHERE f.aircraft_make LIKE '%Learjet%' OR f.aircraft_make LIKE '%Lear%'
       OR r.title LIKE '%Learjet%' OR r.title LIKE '%Lear Jet%'
    ORDER BY f.report_id
""").fetchall()

for rid, title, model in learjet_issues[:15]:
    # Try to extract model from title
    match = re.search(r'Lear\s*[Jj]et[- ]*(\d+[A-Z]?)|LR-?(\d+[A-Z]?)|Model\s+(\d+[A-Z]?)', title, re.IGNORECASE)
    extracted = match.group(1) or match.group(2) or match.group(3) if match else "NOT FOUND"
    status = "OK" if model and len(str(model)) >= 2 else "MISSING"
    print(f"      [{status}] {rid}: model='{model}' | extracted='{extracted}' | {title[:50]}...")

# Cessna issues
print("\n   B) CESSNA MODEL ISSUES:")
cessna_issues = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_make, f.aircraft_model, f.aircraft_category
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
    WHERE r.title LIKE '%Cessna%' OR r.title LIKE '%C-%'
    ORDER BY f.report_id
""").fetchall()

for rid, title, make, model, cat in cessna_issues[:15]:
    match = re.search(r'Cessna[- ]*(\d{3}[A-Z]?)|C-?(\d{3}[A-Z]?)', title, re.IGNORECASE)
    extracted = match.group(1) or match.group(2) if match else "NOT FOUND"
    status = "OK" if cat else "MISSING"
    print(f"      [{status}] {rid}: make='{make}' model='{model}' | extracted='{extracted}' | {title[:50]}...")

# Boeing/DC issues
print("\n   C) BOEING/DOUGLAS CONFUSION:")
boeing_issues = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_make, f.aircraft_model
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
    WHERE f.aircraft_make = 'Boeing'
      AND (r.title LIKE '%DC-%' OR r.title LIKE '%Douglas%' OR r.title LIKE '%McDonnell%')
    ORDER BY f.report_id
""").fetchall()

print(f"      Found {len(boeing_issues)} potential Boeing/Douglas confusion:")
for rid, title, make, model in boeing_issues[:10]:
    print(f"      [{rid}] labeled '{make}' but title: {title[:60]}...")

# 5. Specific patterns we're missing
print("\n\n5. SPECIFIC EXTRACTION EXAMPLES NEEDED")
print("-" * 50)

# Get all titles and find specific patterns
all_titles = cursor.execute("""
    SELECT f.report_id, r.title, f.aircraft_category
    FROM report_features f
    JOIN reports r ON f.report_id = r.filename
""").fetchall()

# Check for specific aircraft we might be missing
specific_checks = [
    (r"Cirrus\s+SR-?\d+", "Cirrus SR"),
    (r"Rolls[- ]?Royce", "Rolls-Royce"),
    (r"Pratt\s*[&and]*\s*Whitney", "Pratt & Whitney"),
    (r"CFM", "CFM engine"),
    (r"Nord\s+\d+", "Nord"),
    (r"NAMC|YS-?\d+", "NAMC YS-11"),
    (r"Aerospatiale|SA-?\d+", "Aerospatiale"),
    (r"Agusta|A-?\d{3}", "Agusta"),
    (r"Bombardier|CL-?\d+|DHC-?\d", "Bombardier/Canadair"),
    (r"Hawker", "Hawker"),
    (r"Pilatus|PC-?\d+", "Pilatus"),
    (r"Shorts?[- ]\d+", "Shorts"),
    (r"CASA|C-?\d{3}", "CASA"),
    (r"ATR[- ]?\d+", "ATR"),
    (r"Fokker|F-?\d{2}", "Fokker"),
]

print("\n   Aircraft types in titles vs extraction status:")
for pattern, name in specific_checks:
    found = []
    for rid, title, cat in all_titles:
        if title and re.search(pattern, title, re.IGNORECASE):
            found.append((rid, title, cat))

    if found:
        extracted = sum(1 for r in found if r[2])
        print(f"\n   {name}: {len(found)} in titles, {extracted} extracted ({extracted/len(found)*100:.0f}%)")
        for rid, title, cat in found[:3]:
            status = "OK" if cat else "MISS"
            print(f"      [{status}] {rid}: {title[:55]}...")

conn.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
