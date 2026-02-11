"""
Aircraft data fetching and categorization.

Data Sources:
1. OpenFlights planes.dat - Open Database License (ODbL)
   URL: https://github.com/jpatokal/openflights/blob/master/data/planes.dat

2. FAA Aircraft Categories - Public Domain (US Government)
   Based on 14 CFR §1.1 definitions

Attribution required for OpenFlights data.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import urllib.request

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/planes.dat"

# Aircraft categorization rules based on manufacturer/model patterns
# These are based on FAA type certificate data and general aviation knowledge
CATEGORIZATION_RULES = {
    # Wide-body jets
    "jet-wide": [
        (r"Boeing\s*7[4678]7", "Boeing wide-body"),
        (r"Boeing\s*777", "Boeing 777"),
        (r"Airbus\s*A3[345]0", "Airbus wide-body"),
        (r"Airbus\s*A380", "Airbus A380"),
        (r"McDonnell\s*Douglas\s*DC-?10", "DC-10"),
        (r"McDonnell\s*Douglas\s*MD-?11", "MD-11"),
        (r"Lockheed\s*L-?1011", "L-1011 Tristar"),
        (r"Douglas\s*DC-?10", "DC-10"),
        (r"DC-?10", "DC-10 generic"),
        (r"MD-?11", "MD-11 generic"),
        (r"747", "Boeing 747"),
        (r"767", "Boeing 767"),
        (r"777", "Boeing 777"),
        (r"787", "Boeing 787"),
        (r"A3[345]0", "Airbus wide"),
        (r"A380", "Airbus A380"),
    ],

    # Narrow-body jets
    "jet-narrow": [
        (r"Boeing\s*70[7-9]", "Boeing 707/720"),
        (r"Boeing\s*72[0-7]", "Boeing 727"),
        (r"Boeing\s*73[0-9]", "Boeing 737"),
        (r"Boeing\s*757", "Boeing 757"),
        (r"Airbus\s*A31[0-9]", "Airbus A310-A319"),
        (r"Airbus\s*A32[0-1]", "Airbus A320/A321"),
        (r"McDonnell\s*Douglas\s*DC-?[89]", "DC-8/DC-9"),
        (r"McDonnell\s*Douglas\s*MD-?8[0-9]", "MD-80 series"),
        (r"McDonnell\s*Douglas\s*MD-?9[0-5]", "MD-90/95"),
        (r"Douglas\s*DC-?[89]", "DC-8/DC-9"),
        (r"DC-?[89]", "DC-8/DC-9 generic"),
        (r"MD-?8[0-9]", "MD-80 generic"),
        (r"737", "Boeing 737"),
        (r"727", "Boeing 727"),
        (r"707", "Boeing 707"),
        (r"757", "Boeing 757"),
        (r"A3[12][0-9]", "Airbus narrow"),
        (r"Fokker\s*(70|100)", "Fokker jet"),
        (r"BAC\s*111", "BAC 1-11"),
        (r"Convair\s*[89]", "Convair jet"),
    ],

    # Regional jets
    "jet-regional": [
        (r"Embraer\s*(E|ERJ)", "Embraer regional jet"),
        (r"CRJ", "Canadair Regional Jet"),
        (r"Canadair.*Regional", "Canadair Regional Jet"),
        (r"Bombardier.*CRJ", "CRJ"),
        (r"Bombardier.*CS", "Bombardier C Series"),
        (r"ERJ", "Embraer ERJ"),
        (r"E-?1[79]0", "Embraer E-Jet"),
        (r"Learjet", "Learjet"),
        (r"Lear\s*\d", "Learjet"),
        (r"Citation", "Cessna Citation"),
        (r"Gulfstream", "Gulfstream"),
        (r"Falcon", "Dassault Falcon"),
        (r"Hawker", "Hawker business jet"),
        (r"BAe\s*146", "BAe 146"),
        (r"Avro\s*RJ", "Avro RJ"),
        (r"Sabre", "Sabreliner"),
    ],

    # Turboprops
    "turboprop": [
        (r"ATR", "ATR"),
        (r"DHC-?[78]", "De Havilland Dash"),
        (r"Dash\s*[78]", "De Havilland Dash"),
        (r"DHC-?6", "DHC-6 Twin Otter"),
        (r"Twin\s*Otter", "Twin Otter"),
        (r"DHC-?3", "DHC-3 Otter"),
        (r"De\s*Havilland.*DHC", "De Havilland"),
        (r"DeHavilland.*DHC", "De Havilland"),
        (r"de\s*Havilland", "De Havilland"),
        (r"Fokker.*F-?27", "Fokker F27"),
        (r"Fokker.*F-?50", "Fokker F50"),
        (r"Saab\s*340", "Saab 340"),
        (r"Saab\s*2000", "Saab 2000"),
        (r"Beech(craft)?\s*1900", "Beech 1900"),
        (r"Beech(craft)?\s*99", "Beech 99"),
        (r"King\s*Air", "King Air"),
        (r"Super\s*King\s*Air", "Super King Air"),
        (r"Shorts?\s*3[36]0", "Shorts 330/360"),
        (r"Jetstream", "BAe Jetstream"),
        (r"Metroliner", "Fairchild Metro"),
        (r"Metro\b", "Fairchild Metro"),
        (r"Swearingen", "Swearingen Metro"),
        (r"Convair\s*[56]", "Convair 580"),
        (r"Lockheed.*Electra", "Lockheed Electra"),
        (r"L-?188", "Lockheed Electra"),
        (r"YS-?11", "NAMC YS-11"),
        (r"Pilatus\s*PC-?12", "Pilatus PC-12"),
        (r"Caravan", "Cessna Caravan"),
        (r"C-?208", "Cessna Caravan"),
        (r"CASA", "CASA"),
        (r"EMB-?110", "Embraer Bandeirante"),
        (r"EMB-?120", "Embraer Brasilia"),
        (r"Bandeirante", "Embraer Bandeirante"),
        (r"Brasilia", "Embraer Brasilia"),
        (r"Heron", "De Havilland Heron"),
        (r"DH-?114", "De Havilland Heron"),
    ],

    # Multi-engine piston
    "multi-piston": [
        (r"Beech(craft)?\s*(Baron|Duke|Queen|Travel)", "Beech multi"),
        (r"Beech(craft)?\s*[56][05]", "Beech multi"),
        (r"Beech(craft)?\s*[6-9]0", "Beech multi"),
        (r"Beech(craft)?\s*[ABCD]-?\d", "Beech multi"),
        (r"Beech(craft)?\s*E-?18", "Beech 18"),
        (r"Beech(craft)?\s*C-?45", "Beech C-45"),
        (r"Cessna\s*3[0-4][0-9]", "Cessna 300 series"),
        (r"Cessna\s*4[0-4][0-9]", "Cessna 400 series"),
        (r"Piper.*Navajo", "Piper Navajo"),
        (r"Piper.*Chieftain", "Piper Chieftain"),
        (r"Piper.*Aztec", "Piper Aztec"),
        (r"Piper.*Seneca", "Piper Seneca"),
        (r"Piper.*PA-?3[12]", "Piper twin"),
        (r"Piper.*PA-?34", "Piper Seneca"),
        (r"Piper.*PA-?23", "Piper Aztec/Apache"),
        (r"Aero\s*Commander", "Aero Commander"),
        (r"Commander\s*[5-9]", "Aero Commander"),
        (r"Douglas\s*DC-?3", "Douglas DC-3"),
        (r"DC-?3", "DC-3"),
        (r"Convair\s*[2-4]", "Convair 240/340/440"),
        (r"Martin\s*[24]0[24]", "Martin 202/404"),
    ],

    # Single-engine piston
    "single-piston": [
        (r"Cessna\s*1[2-8][0-9]", "Cessna singles"),
        (r"Cessna\s*20[0-6]", "Cessna 200 series"),
        (r"Cessna\s*210", "Cessna 210"),
        (r"Piper.*Cherokee", "Piper Cherokee"),
        (r"Piper.*Warrior", "Piper Warrior"),
        (r"Piper.*Archer", "Piper Archer"),
        (r"Piper.*PA-?28", "Piper PA-28"),
        (r"Piper.*PA-?32", "Piper PA-32"),
        (r"Piper.*PA-?24", "Piper Comanche"),
        (r"Piper.*PA-?18", "Piper Cub"),
        (r"Piper.*Cub", "Piper Cub"),
        (r"Beech(craft)?\s*Bonanza", "Beech Bonanza"),
        (r"Beech(craft)?\s*3[35]", "Beech Bonanza/Debonair"),
        (r"Mooney", "Mooney"),
        (r"Cirrus", "Cirrus"),
        (r"Diamond", "Diamond"),
        (r"Grumman.*Tiger", "Grumman Tiger"),
        (r"Grumman.*AA", "Grumman AA series"),
    ],

    # Helicopters
    "helicopter": [
        (r"Sikorsky", "Sikorsky"),
        (r"Bell\s*\d", "Bell"),
        (r"Eurocopter", "Eurocopter"),
        (r"Airbus.*H1[2-7]", "Airbus Helicopters"),
        (r"AS-?3[56]", "Eurocopter AS350/AS365"),
        (r"EC-?1[23]5", "Eurocopter EC"),
        (r"Robinson", "Robinson"),
        (r"R-?[24]4", "Robinson R44"),
        (r"MD\s*Helicopters", "MD Helicopters"),
        (r"Hughes\s*[35]", "Hughes"),
        (r"Agusta", "Agusta"),
        (r"S-?61", "Sikorsky S-61"),
        (r"S-?76", "Sikorsky S-76"),
        (r"UH-?1", "Bell UH-1"),
        (r"BK-?117", "BK 117"),
        (r"Alouette", "Alouette"),
        (r"BO-?105", "MBB BO-105"),
    ],
}


def fetch_openflights_data(cache_path: Optional[Path] = None) -> List[Tuple[str, str, str]]:
    """
    Fetch aircraft data from OpenFlights.

    Returns list of (name, icao_code, iata_code) tuples.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached OpenFlights data from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        print(f"  Fetching OpenFlights data from {OPENFLIGHTS_URL}")
        with urllib.request.urlopen(OPENFLIGHTS_URL) as response:
            content = response.read().decode("utf-8")

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"  Cached to {cache_path}")

    data = []
    for line in content.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 3:
            name = parts[0].strip('"')
            icao = parts[1].strip('"') if parts[1] != "\\N" else None
            iata = parts[2].strip('"') if parts[2] != "\\N" else None
            data.append((name, icao, iata))

    return data


def categorize_aircraft(name: str) -> Tuple[Optional[str], str]:
    """
    Categorize an aircraft based on its name.

    Returns:
        (category, rule_matched) or (None, "no match")
    """
    for category, rules in CATEGORIZATION_RULES.items():
        for pattern, rule_name in rules:
            if re.search(pattern, name, re.IGNORECASE):
                return (category, rule_name)

    return (None, "no match")


def extract_make_model(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract manufacturer and model from aircraft name.

    Examples:
        "Boeing 747-400" -> ("Boeing", "747-400")
        "Cessna 172" -> ("Cessna", "172")
    """
    # Common manufacturer patterns
    manufacturers = [
        (r"^(Boeing)\s+(.+)", "Boeing"),
        (r"^(Airbus)\s+(.+)", "Airbus"),
        (r"^(Cessna)\s+(.+)", "Cessna"),
        (r"^(Piper)\s+(.+)", "Piper"),
        (r"^(Beech(?:craft)?)\s+(.+)", "Beechcraft"),
        (r"^(McDonnell\s*Douglas)\s+(.+)", "McDonnell Douglas"),
        (r"^(Douglas)\s+(.+)", "Douglas"),
        (r"^(Lockheed)\s+(.+)", "Lockheed"),
        (r"^(Embraer)\s+(.+)", "Embraer"),
        (r"^(Bombardier)\s+(.+)", "Bombardier"),
        (r"^(De\s*Havilland|DeHavilland)\s+(.+)", "De Havilland"),
        (r"^(Fokker)\s+(.+)", "Fokker"),
        (r"^(Sikorsky)\s+(.+)", "Sikorsky"),
        (r"^(Bell)\s+(.+)", "Bell"),
        (r"^(Eurocopter)\s+(.+)", "Eurocopter"),
        (r"^(Gulfstream)\s+(.+)", "Gulfstream"),
    ]

    for pattern, make in manufacturers:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            model = match.group(2).strip() if len(match.groups()) > 1 else None
            return (make, model)

    # Fallback: first word is make, rest is model
    parts = name.split(None, 1)
    if len(parts) >= 2:
        return (parts[0], parts[1])
    elif len(parts) == 1:
        return (parts[0], None)

    return (None, None)


def populate_aircraft_lookup(conn, verbose: bool = True):
    """
    Populate aircraft_lookup table from OpenFlights + categorization rules.
    """
    cursor = conn.cursor()

    # Check if already populated
    existing = cursor.execute("SELECT COUNT(*) FROM aircraft_lookup").fetchone()[0]
    if existing > 0:
        if verbose:
            print(f"  aircraft_lookup already has {existing} records.")
        return existing

    # Fetch OpenFlights data
    cache_path = Path("risk_profiler/data/openflights_planes.dat")
    data = fetch_openflights_data(cache_path)

    if verbose:
        print(f"  Processing {len(data)} aircraft types...")

    inserted = 0
    categorized = 0

    for name, icao, iata in data:
        category, rule = categorize_aircraft(name)
        make, model = extract_make_model(name)

        if category:
            categorized += 1

        # Insert with the full name as the pattern (for contains matching)
        cursor.execute("""
            INSERT OR IGNORE INTO aircraft_lookup
            (pattern, pattern_type, make, model, category, icao_code, source, notes)
            VALUES (?, 'contains', ?, ?, ?, ?, 'OpenFlights', ?)
        """, (
            name,
            make or "Unknown",
            model,
            category or "other",
            icao,
            f"Rule: {rule}" if category else "Uncategorized"
        ))

        if cursor.rowcount > 0:
            inserted += 1

    # Add additional patterns from our rules that might not be in OpenFlights
    additional_patterns = [
        # Common abbreviations and variations
        ("DC-3", "Douglas", "DC-3", "multi-piston", "Manual"),
        ("DC-8", "Douglas", "DC-8", "jet-narrow", "Manual"),
        ("DC-9", "Douglas", "DC-9", "jet-narrow", "Manual"),
        ("DC-10", "Douglas", "DC-10", "jet-wide", "Manual"),
        ("MD-80", "McDonnell Douglas", "MD-80", "jet-narrow", "Manual"),
        ("MD-11", "McDonnell Douglas", "MD-11", "jet-wide", "Manual"),
        ("747", "Boeing", "747", "jet-wide", "Manual"),
        ("737", "Boeing", "737", "jet-narrow", "Manual"),
        ("727", "Boeing", "727", "jet-narrow", "Manual"),
        ("707", "Boeing", "707", "jet-narrow", "Manual"),
        ("757", "Boeing", "757", "jet-narrow", "Manual"),
        ("767", "Boeing", "767", "jet-wide", "Manual"),
        ("DHC-6", "De Havilland", "DHC-6", "turboprop", "Manual"),
        ("DHC-8", "De Havilland", "DHC-8", "turboprop", "Manual"),
        ("Twin Otter", "De Havilland", "DHC-6", "turboprop", "Manual"),
        ("King Air", "Beechcraft", "King Air", "turboprop", "Manual"),
        ("Learjet", "Learjet", None, "jet-regional", "Manual"),
        ("Citation", "Cessna", "Citation", "jet-regional", "Manual"),

        # Additional aircraft found in unmatched titles
        ("MU-2", "Mitsubishi", "MU-2", "turboprop", "Manual"),
        ("Mitsubishi", "Mitsubishi", None, "turboprop", "Manual"),
        ("Convair 340", "Convair", "340", "multi-piston", "Manual"),
        ("Convair 440", "Convair", "440", "multi-piston", "Manual"),
        ("Convair 580", "Convair", "580", "turboprop", "Manual"),
        ("Convair 600", "Convair", "600", "turboprop", "Manual"),
        ("CV-580", "Convair", "580", "turboprop", "Manual"),
        ("Falcon", "Dassault", "Falcon", "jet-regional", "Manual"),
        ("Gulfstream", "Gulfstream", None, "jet-regional", "Manual"),
        ("Gulfstream III", "Gulfstream", "III", "jet-regional", "Manual"),
        ("Gulfstream IV", "Gulfstream", "IV", "jet-regional", "Manual"),
        ("Gulfstream V", "Gulfstream", "V", "jet-regional", "Manual"),
        ("G-III", "Gulfstream", "III", "jet-regional", "Manual"),
        ("G-IV", "Gulfstream", "IV", "jet-regional", "Manual"),
        ("G-V", "Gulfstream", "V", "jet-regional", "Manual"),
        ("Cessna 150", "Cessna", "150", "single-piston", "Manual"),
        ("Cessna 152", "Cessna", "152", "single-piston", "Manual"),
        ("PA-28", "Piper", "PA-28", "single-piston", "Manual"),
        ("PA-32", "Piper", "PA-32", "single-piston", "Manual"),
        ("PA-31", "Piper", "PA-31", "multi-piston", "Manual"),
        ("PA-34", "Piper", "PA-34", "multi-piston", "Manual"),
        ("PA-23", "Piper", "PA-23", "multi-piston", "Manual"),
        ("Airbus Helicopters", "Airbus Helicopters", None, "helicopter", "Manual"),
        ("AS350", "Eurocopter", "AS350", "helicopter", "Manual"),
        ("EC135", "Eurocopter", "EC135", "helicopter", "Manual"),
        ("EC145", "Eurocopter", "EC145", "helicopter", "Manual"),
        ("S-76", "Sikorsky", "S-76", "helicopter", "Manual"),
        ("S-61", "Sikorsky", "S-61", "helicopter", "Manual"),
        ("Bell 206", "Bell", "206", "helicopter", "Manual"),
        ("Bell 407", "Bell", "407", "helicopter", "Manual"),
        ("Bell 412", "Bell", "412", "helicopter", "Manual"),
        ("Challenger", "Canadair", "Challenger", "jet-regional", "Manual"),
        ("Canadair", "Canadair", None, "jet-regional", "Manual"),
        ("CL-600", "Canadair", "CL-600", "jet-regional", "Manual"),
        ("ATR 42", "ATR", "42", "turboprop", "Manual"),
        ("ATR 72", "ATR", "72", "turboprop", "Manual"),
        ("ATR-72", "ATR", "72", "turboprop", "Manual"),
        ("Beech 99", "Beechcraft", "99", "turboprop", "Manual"),
        ("Beech 1900", "Beechcraft", "1900", "turboprop", "Manual"),
        ("Beechcraft 1900", "Beechcraft", "1900", "turboprop", "Manual"),
        ("BE-99", "Beechcraft", "99", "turboprop", "Manual"),
        ("BE-1900", "Beechcraft", "1900", "turboprop", "Manual"),
        ("C208", "Cessna", "208", "turboprop", "Manual"),
        ("Cessna 208", "Cessna", "208", "turboprop", "Manual"),
        ("Caravan", "Cessna", "Caravan", "turboprop", "Manual"),
        ("F27", "Fokker", "F27", "turboprop", "Manual"),
        ("F-27", "Fokker", "F27", "turboprop", "Manual"),
        ("Shorts 360", "Shorts", "360", "turboprop", "Manual"),
        ("Shorts 330", "Shorts", "330", "turboprop", "Manual"),
        ("SD3-60", "Shorts", "360", "turboprop", "Manual"),
        ("HS 125", "Hawker Siddeley", "125", "jet-regional", "Manual"),
        ("HS-125", "Hawker Siddeley", "125", "jet-regional", "Manual"),
        ("BAe 125", "BAe", "125", "jet-regional", "Manual"),
        ("Hawker 800", "Hawker", "800", "jet-regional", "Manual"),
        ("Jetstream", "BAe", "Jetstream", "turboprop", "Manual"),
        ("Metro", "Fairchild", "Metro", "turboprop", "Manual"),
        ("SA-226", "Swearingen", "SA-226", "turboprop", "Manual"),
        ("SA-227", "Swearingen", "SA-227", "turboprop", "Manual"),
        ("Swearingen", "Swearingen", None, "turboprop", "Manual"),
        ("Fairchild", "Fairchild", None, "turboprop", "Manual"),
        ("Pilatus PC-12", "Pilatus", "PC-12", "turboprop", "Manual"),
        ("PC-12", "Pilatus", "PC-12", "turboprop", "Manual"),
        ("L-188", "Lockheed", "L-188", "turboprop", "Manual"),
        ("Electra", "Lockheed", "Electra", "turboprop", "Manual"),
        ("Grumman", "Grumman", None, "turboprop", "Manual"),
        ("G-73", "Grumman", "G-73", "turboprop", "Manual"),
        ("Mallard", "Grumman", "Mallard", "turboprop", "Manual"),

        # Additional patterns from unmatched titles analysis
        ("DH-114", "De Havilland", "DH-114", "multi-piston", "Manual"),
        ("Heron", "De Havilland", "Heron", "multi-piston", "Manual"),
        ("PRINAIR", "De Havilland", "DH-114", "multi-piston", "Manual"),  # Puerto Rico airline flew Herons

        # EMB series (Embraer regional)
        ("EMB-120", "Embraer", "EMB-120", "turboprop", "Manual"),
        ("EMB-110", "Embraer", "EMB-110", "turboprop", "Manual"),
        ("EMB 120", "Embraer", "EMB-120", "turboprop", "Manual"),
        ("Comair", "Embraer", "EMB-120", "turboprop", "Manual"),  # Common operator

        # Beech patterns
        ("B99", "Beechcraft", "99", "turboprop", "Manual"),
        ("B-99", "Beechcraft", "99", "turboprop", "Manual"),
        ("Beech 99", "Beechcraft", "99", "turboprop", "Manual"),

        # Sabreliner variants
        ("Sabre 40", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Sabre 60", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Sabre 65", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Sabre 75", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Sabre Mark", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Sabreliner", "North American", "Sabreliner", "jet-regional", "Manual"),

        # Twin Commander
        ("Twin Commander", "Aero Commander", "Twin Commander", "multi-piston", "Manual"),
        ("Commander 500", "Aero Commander", "500", "multi-piston", "Manual"),
        ("Commander 680", "Aero Commander", "680", "multi-piston", "Manual"),
        ("Commander 690", "Aero Commander", "690", "turboprop", "Manual"),

        # Additional Boeing patterns for airlines
        ("Southwest Airlines", "Boeing", "737", "jet-narrow", "Manual"),  # Southwest only flies 737s
        ("American Airlines", "Boeing", None, "jet-narrow", "Manual"),  # Most AA narrowbody
        ("Delta Air Lines", "Boeing", None, "jet-narrow", "Manual"),
        ("United Airlines", "Boeing", None, "jet-narrow", "Manual"),
        ("USAir", "Boeing", None, "jet-narrow", "Manual"),
        ("US Airways", "Boeing", None, "jet-narrow", "Manual"),

        # Additional turboprops
        ("Casa 212", "CASA", "212", "turboprop", "Manual"),
        ("CASA 212", "CASA", "212", "turboprop", "Manual"),
        ("Dash 7", "De Havilland", "DHC-7", "turboprop", "Manual"),
        ("Dash 8", "De Havilland", "DHC-8", "turboprop", "Manual"),
        ("Dash-8", "De Havilland", "DHC-8", "turboprop", "Manual"),

        # Additional helicopters
        ("MBB", "MBB", "BO-105", "helicopter", "Manual"),
        ("BO 105", "MBB", "BO-105", "helicopter", "Manual"),
        ("BK 117", "MBB", "BK-117", "helicopter", "Manual"),

        # Cessna Caravans and variants
        ("Grand Caravan", "Cessna", "208B", "turboprop", "Manual"),
        ("Cessna Caravan", "Cessna", "208", "turboprop", "Manual"),

        # Douglas DC series
        ("Douglas DC-3", "Douglas", "DC-3", "multi-piston", "Manual"),
        ("Douglas DC-6", "Douglas", "DC-6", "multi-piston", "Manual"),
        ("Douglas DC-7", "Douglas", "DC-7", "multi-piston", "Manual"),
        ("DC-6", "Douglas", "DC-6", "multi-piston", "Manual"),
        ("DC-7", "Douglas", "DC-7", "multi-piston", "Manual"),

        # Misc patterns from titles
        ("Spectrum Air", "North American", "Sabreliner", "jet-regional", "Manual"),
        ("Friday Harbor", "De Havilland", "DHC-3", "turboprop", "Manual"),  # Floatplane operator
        ("West Isle Air", "De Havilland", "DHC-3", "turboprop", "Manual"),

        # More Cessna twins (400 series)
        ("Cessna 401", "Cessna", "401", "multi-piston", "Manual"),
        ("Cessna 402", "Cessna", "402", "multi-piston", "Manual"),
        ("Cessna 404", "Cessna", "404", "multi-piston", "Manual"),
        ("Cessna 411", "Cessna", "411", "multi-piston", "Manual"),
        ("Cessna 414", "Cessna", "414", "multi-piston", "Manual"),
        ("Cessna 421", "Cessna", "421", "multi-piston", "Manual"),
        ("Cessna 425", "Cessna", "425", "turboprop", "Manual"),
        ("Cessna 441", "Cessna", "441", "turboprop", "Manual"),

        # Beech C-99 variant
        ("C-99", "Beechcraft", "C99", "turboprop", "Manual"),
        ("C99", "Beechcraft", "C99", "turboprop", "Manual"),

        # Lockheed L-1011 TriStar
        ("L-1011", "Lockheed", "L-1011", "jet-wide", "Manual"),
        ("L1011", "Lockheed", "L-1011", "jet-wide", "Manual"),
        ("TriStar", "Lockheed", "L-1011", "jet-wide", "Manual"),

        # Convair variants
        ("Convair 240", "Convair", "240", "multi-piston", "Manual"),
        ("Convair 340", "Convair", "340", "multi-piston", "Manual"),
        ("Convair 440", "Convair", "440", "multi-piston", "Manual"),
        ("CV-240", "Convair", "240", "multi-piston", "Manual"),
        ("CV-340", "Convair", "340", "multi-piston", "Manual"),
        ("CV-440", "Convair", "440", "multi-piston", "Manual"),

        # Canadair CL-44 (cargo)
        ("CL-44", "Canadair", "CL-44", "turboprop", "Manual"),

        # Fairchild variants
        ("FH-227", "Fairchild", "FH-227", "turboprop", "Manual"),
        ("Fairchild Hiller", "Fairchild", None, "turboprop", "Manual"),

        # De Havilland DHC-2 Beaver (floatplane)
        ("DHC-2", "De Havilland", "DHC-2", "single-piston", "Manual"),
        ("Beaver", "De Havilland", "DHC-2", "single-piston", "Manual"),

        # Cessna 210 singles
        ("Cessna 210", "Cessna", "210", "single-piston", "Manual"),
        ("C-210", "Cessna", "210", "single-piston", "Manual"),

        # Military/historic aircraft
        ("B-17", "Boeing", "B-17", "multi-piston", "Manual"),
        ("B-52", "Boeing", "B-52", "jet-wide", "Manual"),
        ("C-130", "Lockheed", "C-130", "turboprop", "Manual"),
        ("C-141", "Lockheed", "C-141", "jet-wide", "Manual"),
        ("P-51", "North American", "P-51", "single-piston", "Manual"),

        # Peninsula Aviation
        ("Peninsula Aviation", "De Havilland", "DHC-3", "turboprop", "Manual"),
    ]

    for pattern, make, model, category, source in additional_patterns:
        cursor.execute("""
            INSERT OR IGNORE INTO aircraft_lookup
            (pattern, pattern_type, make, model, category, source, notes)
            VALUES (?, 'contains', ?, ?, ?, ?, 'Common abbreviation')
        """, (pattern, make, model, category, source))
        if cursor.rowcount > 0:
            inserted += 1

    conn.commit()

    if verbose:
        print(f"  Inserted {inserted} records")
        print(f"  Categorized {categorized}/{len(data)} OpenFlights aircraft")

    return inserted


def fuzzy_match_aircraft(text: str, all_patterns: list, threshold: float = 0.85) -> Optional[tuple]:
    """
    Fuzzy match text against known aircraft patterns.

    Uses simple character-based similarity (Jaccard on character trigrams).
    Returns (pattern, similarity) or None if no match above threshold.
    """
    def trigrams(s):
        s = s.lower()
        return set(s[i:i+3] for i in range(len(s)-2)) if len(s) >= 3 else {s.lower()}

    def jaccard(a, b):
        if not a or not b:
            return 0
        return len(a & b) / len(a | b)

    text_upper = text.upper()
    best_match = None
    best_score = 0

    for pattern, make, model, category, source in all_patterns:
        # Skip very short patterns for fuzzy matching
        if len(pattern) < 4:
            continue

        # Check if pattern words appear in text (word-level fuzzy)
        pattern_upper = pattern.upper()
        words = pattern_upper.split()

        # For multi-word patterns, check if words appear near each other
        if len(words) > 1:
            all_found = all(w in text_upper for w in words)
            if all_found:
                score = 0.95
                if score > best_score:
                    best_score = score
                    best_match = (pattern, make, model, category, source)
                continue

        # Trigram similarity for single words or partial matches
        pattern_tris = trigrams(pattern)
        for i in range(len(text) - len(pattern) + 1):
            window = text[i:i+len(pattern)+3]
            window_tris = trigrams(window)
            score = jaccard(pattern_tris, window_tris)
            if score > best_score:
                best_score = score
                best_match = (pattern, make, model, category, source)

    if best_score >= threshold and best_match:
        return best_match, best_score
    return None


def lookup_aircraft(conn, text: str, use_fuzzy: bool = True) -> Optional[Dict]:
    """
    Look up aircraft category from text (e.g., title field).

    Uses case-insensitive matching with UPPER() for robustness.
    Falls back to fuzzy matching if enabled.
    Returns dict with make, model, category, confidence or None.
    """
    cursor = conn.cursor()

    # Try to find a matching pattern (case-insensitive with UPPER)
    result = cursor.execute("""
        SELECT pattern, make, model, category, source
        FROM aircraft_lookup
        WHERE UPPER(?) LIKE '%' || UPPER(pattern) || '%'
        ORDER BY LENGTH(pattern) DESC
        LIMIT 1
    """, (text,)).fetchone()

    if result:
        return {
            "pattern_matched": result[0],
            "make": result[1],
            "model": result[2],
            "category": result[3],
            "source": result[4],
            "confidence": "high" if result[4] == "OpenFlights" else "medium"
        }

    # Try fuzzy matching as fallback
    if use_fuzzy and text:
        all_patterns = cursor.execute("""
            SELECT pattern, make, model, category, source
            FROM aircraft_lookup
            WHERE LENGTH(pattern) >= 4
            ORDER BY LENGTH(pattern) DESC
        """).fetchall()

        fuzzy_result = fuzzy_match_aircraft(text, all_patterns, threshold=0.85)
        if fuzzy_result:
            match, score = fuzzy_result
            return {
                "pattern_matched": match[0],
                "make": match[1],
                "model": match[2],
                "category": match[3],
                "source": match[4],
                "confidence": "low",  # Fuzzy matches are lower confidence
                "fuzzy_score": score
            }

    return None


if __name__ == "__main__":
    # Test the module
    import sqlite3

    conn = sqlite3.connect("sqlite/riskradar.db")
    populate_aircraft_lookup(conn, verbose=True)

    # Test some lookups
    test_titles = [
        "Crash of Boeing 747-121, N123AB",
        "Cessna 172 collision with terrain",
        "McDonnell Douglas DC-9-31, N456XY",
        "DHC-6 Twin Otter accident",
        "Bell 206 helicopter crash",
    ]

    print("\nTest lookups:")
    for title in test_titles:
        result = lookup_aircraft(conn, title)
        if result:
            print(f"  '{title[:40]}...' -> {result['category']} ({result['make']} {result['model']})")
        else:
            print(f"  '{title[:40]}...' -> NO MATCH")

    conn.close()
