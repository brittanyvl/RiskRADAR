"""
Additional aircraft patterns to add to the database.

These cover gaps found in the deep analysis:
1. Dirty text (DC8 vs DC-8)
2. Missing manufacturers (Short Brothers, Britten-Norman)
3. Specific Cessna models
4. Beechcraft variants
5. Other gaps identified in validation
"""

# Format: (pattern, make, model, category, source)
ADDITIONAL_PATTERNS = [
    # =========================================
    # DIRTY TEXT VARIATIONS (no hyphen)
    # =========================================
    ("DC8", "Douglas", "DC-8", "jet-narrow", "Manual"),
    ("DC9", "Douglas", "DC-9", "jet-narrow", "Manual"),
    ("DC10", "Douglas", "DC-10", "jet-wide", "Manual"),
    ("MD80", "McDonnell Douglas", "MD-80", "jet-narrow", "Manual"),
    ("MD81", "McDonnell Douglas", "MD-81", "jet-narrow", "Manual"),
    ("MD82", "McDonnell Douglas", "MD-82", "jet-narrow", "Manual"),
    ("MD83", "McDonnell Douglas", "MD-83", "jet-narrow", "Manual"),
    ("MD87", "McDonnell Douglas", "MD-87", "jet-narrow", "Manual"),
    ("MD88", "McDonnell Douglas", "MD-88", "jet-narrow", "Manual"),
    ("MD90", "McDonnell Douglas", "MD-90", "jet-narrow", "Manual"),
    ("MD11", "McDonnell Douglas", "MD-11", "jet-wide", "Manual"),

    # =========================================
    # SHORT BROTHERS / SHORTS
    # =========================================
    ("Short Brothers", "Short Brothers", "Skyvan", "turboprop", "Manual"),
    ("Skyvan", "Short Brothers", "Skyvan", "turboprop", "Manual"),
    ("Shorts 330", "Short Brothers", "330", "turboprop", "Manual"),
    ("Shorts 360", "Short Brothers", "360", "turboprop", "Manual"),
    ("SD3-30", "Short Brothers", "SD3-30", "turboprop", "Manual"),
    ("SD3-60", "Short Brothers", "SD3-60", "turboprop", "Manual"),
    ("Sherpa", "Short Brothers", "Sherpa", "turboprop", "Manual"),

    # =========================================
    # BRITTEN-NORMAN
    # =========================================
    ("Britten-Norman", "Britten-Norman", "Islander", "multi-piston", "Manual"),
    ("Britten Norman", "Britten-Norman", "Islander", "multi-piston", "Manual"),
    ("BN-2", "Britten-Norman", "Islander", "multi-piston", "Manual"),
    ("Islander", "Britten-Norman", "Islander", "multi-piston", "Manual"),
    ("Trislander", "Britten-Norman", "Trislander", "multi-piston", "Manual"),

    # =========================================
    # CESSNA SINGLE-ENGINE (100-200 series)
    # =========================================
    ("Cessna 150", "Cessna", "150", "single-piston", "Manual"),
    ("Cessna 152", "Cessna", "152", "single-piston", "Manual"),
    ("Cessna 170", "Cessna", "170", "single-piston", "Manual"),
    ("Cessna 172", "Cessna", "172", "single-piston", "Manual"),
    ("Cessna 175", "Cessna", "175", "single-piston", "Manual"),
    ("Cessna 177", "Cessna", "177", "single-piston", "Manual"),
    ("Cessna 180", "Cessna", "180", "single-piston", "Manual"),
    ("Cessna 182", "Cessna", "182", "single-piston", "Manual"),
    ("Cessna 185", "Cessna", "185", "single-piston", "Manual"),
    ("Cessna 188", "Cessna", "188", "single-piston", "Manual"),
    ("Cessna 190", "Cessna", "190", "single-piston", "Manual"),
    ("Cessna 195", "Cessna", "195", "single-piston", "Manual"),
    ("Cessna 205", "Cessna", "205", "single-piston", "Manual"),
    ("Cessna 206", "Cessna", "206", "single-piston", "Manual"),
    ("Cessna 207", "Cessna", "207", "single-piston", "Manual"),
    ("Cessna 210", "Cessna", "210", "single-piston", "Manual"),

    # Cessna with letter prefixes
    ("Cessna P210", "Cessna", "P210", "single-piston", "Manual"),
    ("Cessna T210", "Cessna", "T210", "single-piston", "Manual"),
    ("Cessna U206", "Cessna", "U206", "single-piston", "Manual"),
    ("Cessna TU206", "Cessna", "TU206", "single-piston", "Manual"),

    # =========================================
    # CESSNA TWINS (300-400 series)
    # =========================================
    ("Cessna 303", "Cessna", "303", "multi-piston", "Manual"),
    ("Cessna 310", "Cessna", "310", "multi-piston", "Manual"),
    ("Cessna 310R", "Cessna", "310R", "multi-piston", "Manual"),
    ("Cessna 320", "Cessna", "320", "multi-piston", "Manual"),
    ("Cessna 335", "Cessna", "335", "multi-piston", "Manual"),
    ("Cessna 336", "Cessna", "336", "multi-piston", "Manual"),
    ("Cessna 337", "Cessna", "337", "multi-piston", "Manual"),
    ("Cessna 340", "Cessna", "340", "multi-piston", "Manual"),
    ("Cessna 340A", "Cessna", "340A", "multi-piston", "Manual"),
    ("Cessna 401", "Cessna", "401", "multi-piston", "Manual"),
    ("Cessna 402", "Cessna", "402", "multi-piston", "Manual"),
    ("Cessna 402C", "Cessna", "402C", "multi-piston", "Manual"),
    ("Cessna 404", "Cessna", "404", "multi-piston", "Manual"),
    ("Cessna 411", "Cessna", "411", "multi-piston", "Manual"),
    ("Cessna 414", "Cessna", "414", "multi-piston", "Manual"),
    ("Cessna 414A", "Cessna", "414A", "multi-piston", "Manual"),
    ("Cessna 421", "Cessna", "421", "multi-piston", "Manual"),
    ("Cessna-421", "Cessna", "421", "multi-piston", "Manual"),
    ("Cessna 425", "Cessna", "425", "turboprop", "Manual"),
    ("Cessna 441", "Cessna", "441", "turboprop", "Manual"),

    # =========================================
    # CESSNA CITATION (jets)
    # =========================================
    ("Cessna 500", "Cessna", "Citation 500", "jet-regional", "Manual"),
    ("Cessna 501", "Cessna", "Citation 501", "jet-regional", "Manual"),
    ("Cessna 525", "Cessna", "Citation CJ1", "jet-regional", "Manual"),
    ("Cessna 550", "Cessna", "Citation II", "jet-regional", "Manual"),
    ("Cessna 551", "Cessna", "Citation II/SP", "jet-regional", "Manual"),
    ("Cessna 560", "Cessna", "Citation V", "jet-regional", "Manual"),
    ("Cessna 650", "Cessna", "Citation III", "jet-regional", "Manual"),
    ("Cessna 680", "Cessna", "Citation Sovereign", "jet-regional", "Manual"),
    ("Cessna 750", "Cessna", "Citation X", "jet-regional", "Manual"),
    ("Citation 500", "Cessna", "Citation 500", "jet-regional", "Manual"),
    ("Citation 501", "Cessna", "Citation 501", "jet-regional", "Manual"),
    ("Citation I", "Cessna", "Citation I", "jet-regional", "Manual"),
    ("Citation II", "Cessna", "Citation II", "jet-regional", "Manual"),
    ("Citation III", "Cessna", "Citation III", "jet-regional", "Manual"),
    ("Citation V", "Cessna", "Citation V", "jet-regional", "Manual"),
    ("Citation X", "Cessna", "Citation X", "jet-regional", "Manual"),

    # =========================================
    # BEECHCRAFT VARIANTS
    # =========================================
    # Beech twins (multi-piston)
    ("Beech 18", "Beechcraft", "18", "multi-piston", "Manual"),
    ("Beech E18S", "Beechcraft", "E18S", "multi-piston", "Manual"),
    ("Beech D18S", "Beechcraft", "D18S", "multi-piston", "Manual"),
    ("Beech G18S", "Beechcraft", "G18S", "multi-piston", "Manual"),
    ("Beech C-45", "Beechcraft", "C-45", "multi-piston", "Manual"),
    ("Beechcraft 18", "Beechcraft", "18", "multi-piston", "Manual"),
    ("Beechcraft 50", "Beechcraft", "50", "multi-piston", "Manual"),
    ("Beechcraft 55", "Beechcraft", "55", "multi-piston", "Manual"),
    ("Beechcraft 56", "Beechcraft", "56", "multi-piston", "Manual"),
    ("Beechcraft 58", "Beechcraft", "58", "multi-piston", "Manual"),
    ("Beechcraft 60", "Beechcraft", "60", "multi-piston", "Manual"),
    ("Beechcraft 65", "Beechcraft", "65", "multi-piston", "Manual"),
    ("Beech 65-B80", "Beechcraft", "65-B80", "multi-piston", "Manual"),
    ("Beechcraft 65-B80", "Beechcraft", "65-B80", "multi-piston", "Manual"),
    ("Beech 65-A80", "Beechcraft", "65-A80", "multi-piston", "Manual"),
    ("Beechcraft 65-A80", "Beechcraft", "65-A80", "multi-piston", "Manual"),
    ("Beech 70", "Beechcraft", "70", "multi-piston", "Manual"),
    ("Beechcraft 70", "Beechcraft", "70", "multi-piston", "Manual"),
    ("Queen Air", "Beechcraft", "Queen Air", "multi-piston", "Manual"),
    ("Baron", "Beechcraft", "Baron", "multi-piston", "Manual"),
    ("Baron 55", "Beechcraft", "Baron 55", "multi-piston", "Manual"),
    ("Baron 58", "Beechcraft", "Baron 58", "multi-piston", "Manual"),
    ("D-55 Baron", "Beechcraft", "D-55 Baron", "multi-piston", "Manual"),

    # Beech King Air (turboprop)
    ("King Air", "Beechcraft", "King Air", "turboprop", "Manual"),
    ("King Air 90", "Beechcraft", "King Air 90", "turboprop", "Manual"),
    ("King Air 100", "Beechcraft", "King Air 100", "turboprop", "Manual"),
    ("King Air 200", "Beechcraft", "King Air 200", "turboprop", "Manual"),
    ("King Air 300", "Beechcraft", "King Air 300", "turboprop", "Manual"),
    ("King Air 350", "Beechcraft", "King Air 350", "turboprop", "Manual"),
    ("Super King Air", "Beechcraft", "Super King Air", "turboprop", "Manual"),
    ("Beechcraft A100", "Beechcraft", "A100", "turboprop", "Manual"),
    ("Beech A100", "Beechcraft", "A100", "turboprop", "Manual"),
    ("Beech AI00", "Beechcraft", "A100", "turboprop", "Manual"),  # Typo variant

    # Beech 99/1900 (turboprop)
    ("Beech 99", "Beechcraft", "99", "turboprop", "Manual"),
    ("Beech B99", "Beechcraft", "B99", "turboprop", "Manual"),
    ("Beech 99A", "Beechcraft", "99A", "turboprop", "Manual"),
    ("Beechcraft 99", "Beechcraft", "99", "turboprop", "Manual"),
    ("Beechcraft 99A", "Beechcraft", "99A", "turboprop", "Manual"),
    ("BE-99", "Beechcraft", "99", "turboprop", "Manual"),
    ("Beech 1900", "Beechcraft", "1900", "turboprop", "Manual"),
    ("Beech 1900C", "Beechcraft", "1900C", "turboprop", "Manual"),
    ("Beech 1900D", "Beechcraft", "1900D", "turboprop", "Manual"),
    ("Beechcraft 1900", "Beechcraft", "1900", "turboprop", "Manual"),
    ("Raytheon Beechcraft 1900", "Beechcraft", "1900", "turboprop", "Manual"),
    ("Raytheon 1900", "Beechcraft", "1900", "turboprop", "Manual"),

    # Beechjet (jet-regional)
    ("Beechjet", "Beechcraft", "Beechjet", "jet-regional", "Manual"),
    ("Beech-Hawker", "Hawker Beechcraft", "125", "jet-regional", "Manual"),
    ("Hawker Beechcraft", "Hawker Beechcraft", "125", "jet-regional", "Manual"),

    # =========================================
    # LEARJET (more models)
    # =========================================
    ("Learjet 23", "Learjet", "23", "jet-regional", "Manual"),
    ("Learjet 24", "Learjet", "24", "jet-regional", "Manual"),
    ("Learjet 24B", "Learjet", "24B", "jet-regional", "Manual"),
    ("Learjet 24D", "Learjet", "24D", "jet-regional", "Manual"),
    ("Learjet 25", "Learjet", "25", "jet-regional", "Manual"),
    ("Learjet 25B", "Learjet", "25B", "jet-regional", "Manual"),
    ("Learjet 25C", "Learjet", "25C", "jet-regional", "Manual"),
    ("Learjet 25D", "Learjet", "25D", "jet-regional", "Manual"),
    ("Learjet 28", "Learjet", "28", "jet-regional", "Manual"),
    ("Learjet 31", "Learjet", "31", "jet-regional", "Manual"),
    ("Learjet 35", "Learjet", "35", "jet-regional", "Manual"),
    ("Learjet 35A", "Learjet", "35A", "jet-regional", "Manual"),
    ("Learjet 36", "Learjet", "36", "jet-regional", "Manual"),
    ("Learjet 45", "Learjet", "45", "jet-regional", "Manual"),
    ("Learjet 55", "Learjet", "55", "jet-regional", "Manual"),
    ("Learjet 60", "Learjet", "60", "jet-regional", "Manual"),
    ("Lear Jet 23", "Learjet", "23", "jet-regional", "Manual"),
    ("Lear Jet 24", "Learjet", "24", "jet-regional", "Manual"),
    ("Lear Jet 25", "Learjet", "25", "jet-regional", "Manual"),
    ("Lear Jet L23A", "Learjet", "L23A", "jet-regional", "Manual"),
    ("Gates Learjet", "Learjet", None, "jet-regional", "Manual"),
    ("LR24", "Learjet", "24", "jet-regional", "Manual"),
    ("LR24B", "Learjet", "24B", "jet-regional", "Manual"),
    ("LR25", "Learjet", "25", "jet-regional", "Manual"),

    # =========================================
    # SABRELINER / ROCKWELL
    # =========================================
    ("Sabreliner", "Rockwell", "Sabreliner", "jet-regional", "Manual"),
    ("Sabre Model", "Rockwell", "Sabreliner", "jet-regional", "Manual"),
    ("Sabre Mark 5", "Rockwell", "Sabreliner Mark 5", "jet-regional", "Manual"),
    ("NA-265", "North American", "Sabreliner", "jet-regional", "Manual"),
    ("NA-265-60", "North American", "Sabreliner 60", "jet-regional", "Manual"),
    ("NA-265-65", "North American", "Sabreliner 65", "jet-regional", "Manual"),

    # =========================================
    # FAIRCHILD / METRO
    # =========================================
    ("Fairchild Hiller", "Fairchild Hiller", "FH-227", "turboprop", "Manual"),
    ("FH-227", "Fairchild Hiller", "FH-227", "turboprop", "Manual"),
    ("FH-227B", "Fairchild Hiller", "FH-227B", "turboprop", "Manual"),
    ("FH-227C", "Fairchild Hiller", "FH-227C", "turboprop", "Manual"),
    ("Fairchild F-27", "Fairchild", "F-27", "turboprop", "Manual"),
    ("F-27B", "Fairchild", "F-27B", "turboprop", "Manual"),
    ("Metro II", "Fairchild Swearingen", "Metro II", "turboprop", "Manual"),
    ("Metro III", "Fairchild Swearingen", "Metro III", "turboprop", "Manual"),
    ("SA-226", "Swearingen", "SA-226", "turboprop", "Manual"),
    ("SA-227", "Swearingen", "SA-227", "turboprop", "Manual"),
    ("SA227", "Swearingen", "SA-227", "turboprop", "Manual"),

    # =========================================
    # SCALED COMPOSITES / EXPERIMENTAL
    # =========================================
    ("Scaled Composites", "Scaled Composites", "SpaceShipTwo", "other", "Manual"),
    ("SpaceShipTwo", "Scaled Composites", "SpaceShipTwo", "other", "Manual"),
    ("SpaceShip", "Scaled Composites", "SpaceShipTwo", "other", "Manual"),

    # =========================================
    # GLIDERS
    # =========================================
    ("Schempp-Hirth", "Schempp-Hirth", "Nimbus", "other", "Manual"),
    ("Nimbus", "Schempp-Hirth", "Nimbus", "other", "Manual"),

    # =========================================
    # LOCKHEED
    # =========================================
    ("Lockheed Jetstar", "Lockheed", "Jetstar", "jet-regional", "Manual"),
    ("Jetstar", "Lockheed", "Jetstar", "jet-regional", "Manual"),
    ("Learstar", "Lockheed", "Learstar", "multi-piston", "Manual"),
    ("L-18", "Lockheed", "L-18", "multi-piston", "Manual"),
    ("L-188", "Lockheed", "L-188", "turboprop", "Manual"),
    ("L-382", "Lockheed", "L-382", "turboprop", "Manual"),
    ("L-382G", "Lockheed", "L-382G", "turboprop", "Manual"),
    ("L-1011", "Lockheed", "L-1011", "jet-wide", "Manual"),
    ("L-1049", "Lockheed", "L-1049", "multi-piston", "Manual"),
    ("L-1049H", "Lockheed", "L-1049H", "multi-piston", "Manual"),
    ("Super Constellation", "Lockheed", "Super Constellation", "multi-piston", "Manual"),
    ("Constellation", "Lockheed", "Constellation", "multi-piston", "Manual"),

    # =========================================
    # GULFSTREAM
    # =========================================
    ("Gulfstream II", "Grumman", "Gulfstream II", "jet-regional", "Manual"),
    ("Gulfstream III", "Gulfstream", "III", "jet-regional", "Manual"),
    ("Gulfstream IV", "Gulfstream", "IV", "jet-regional", "Manual"),
    ("Gulfstream V", "Gulfstream", "V", "jet-regional", "Manual"),
    ("Gulfstream G-IV", "Gulfstream", "G-IV", "jet-regional", "Manual"),
    ("Gulfstream G-V", "Gulfstream", "G-V", "jet-regional", "Manual"),
    ("Gulfstream G650", "Gulfstream", "G650", "jet-regional", "Manual"),
    ("G-II", "Gulfstream", "II", "jet-regional", "Manual"),
    ("G-III", "Gulfstream", "III", "jet-regional", "Manual"),
    ("G-IV", "Gulfstream", "IV", "jet-regional", "Manual"),
    ("G-V", "Gulfstream", "V", "jet-regional", "Manual"),
    ("GVI", "Gulfstream", "G650", "jet-regional", "Manual"),

    # =========================================
    # GRUMMAN (non-Gulfstream)
    # =========================================
    ("Grumman G21", "Grumman", "G21 Goose", "multi-piston", "Manual"),
    ("Grumman G21A", "Grumman", "G21A Goose", "multi-piston", "Manual"),
    ("Grumman G73", "Grumman", "G73 Mallard", "multi-piston", "Manual"),
    ("Grumman G73T", "Grumman", "G73T Turbo Mallard", "turboprop", "Manual"),
    ("Turbo Mallard", "Grumman", "G73T Turbo Mallard", "turboprop", "Manual"),
    ("Mallard", "Grumman", "Mallard", "multi-piston", "Manual"),
    ("Goose", "Grumman", "Goose", "multi-piston", "Manual"),
    ("G-1159", "Grumman", "G-1159", "jet-regional", "Manual"),

    # =========================================
    # DE HAVILLAND CANADA
    # =========================================
    ("De Havilland Canada", "De Havilland Canada", "DHC", "turboprop", "Manual"),
    ("de Havilland Canada", "De Havilland Canada", "DHC", "turboprop", "Manual"),
    ("DHC-2", "De Havilland Canada", "DHC-2 Beaver", "single-piston", "Manual"),
    ("DHC-3", "De Havilland Canada", "DHC-3 Otter", "single-piston", "Manual"),
    ("DHC-3T", "De Havilland Canada", "DHC-3T Turbo Otter", "turboprop", "Manual"),
    ("DHC-4", "De Havilland Canada", "DHC-4 Caribou", "turboprop", "Manual"),
    ("DHC-5", "De Havilland Canada", "DHC-5 Buffalo", "turboprop", "Manual"),
    ("DHC-6", "De Havilland Canada", "DHC-6 Twin Otter", "turboprop", "Manual"),
    ("DHC-7", "De Havilland Canada", "DHC-7 Dash 7", "turboprop", "Manual"),
    ("DHC-8", "De Havilland Canada", "DHC-8 Dash 8", "turboprop", "Manual"),
    ("DHC 8", "De Havilland Canada", "DHC-8", "turboprop", "Manual"),
    ("Beaver", "De Havilland Canada", "DHC-2 Beaver", "single-piston", "Manual"),
    ("Otter", "De Havilland Canada", "DHC-3 Otter", "single-piston", "Manual"),
    ("Twin Otter", "De Havilland Canada", "DHC-6 Twin Otter", "turboprop", "Manual"),
    ("Dash 7", "De Havilland Canada", "DHC-7 Dash 7", "turboprop", "Manual"),
    ("Dash 8", "De Havilland Canada", "DHC-8 Dash 8", "turboprop", "Manual"),
    ("Bombardier DHC", "De Havilland Canada", "DHC", "turboprop", "Manual"),

    # De Havilland UK
    ("DH-104", "De Havilland", "DH-104 Dove", "multi-piston", "Manual"),
    ("DH-114", "De Havilland", "DH-114 Heron", "multi-piston", "Manual"),
    ("Dove", "De Havilland", "DH-104 Dove", "multi-piston", "Manual"),
    ("Heron", "De Havilland", "DH-114 Heron", "multi-piston", "Manual"),

    # =========================================
    # HELICOPTERS - ADDITIONAL
    # =========================================
    ("Sikorsky S-61", "Sikorsky", "S-61", "helicopter", "Manual"),
    ("S-61L", "Sikorsky", "S-61L", "helicopter", "Manual"),
    ("5-61L", "Sikorsky", "S-61L", "helicopter", "Manual"),  # Typo variant
    ("Bell UH-1", "Bell", "UH-1", "helicopter", "Manual"),
    ("Bell UG-1B", "Bell", "UH-1B", "helicopter", "Manual"),  # Typo variant
    ("Bell UH-1B", "Bell", "UH-1B", "helicopter", "Manual"),
    ("EC-135", "Eurocopter", "EC-135", "helicopter", "Manual"),
    ("EC135", "Eurocopter", "EC-135", "helicopter", "Manual"),
    ("AS350", "Eurocopter", "AS350", "helicopter", "Manual"),
    ("AS-350", "Eurocopter", "AS350", "helicopter", "Manual"),
    ("AS350 B2", "Eurocopter", "AS350 B2", "helicopter", "Manual"),
    ("AS350 B3", "Eurocopter", "AS350 B3", "helicopter", "Manual"),
    ("AS350-B2", "Eurocopter", "AS350 B2", "helicopter", "Manual"),
    ("SA365", "Aerospatiale", "SA365", "helicopter", "Manual"),
    ("SA365N1", "Aerospatiale", "SA365N1", "helicopter", "Manual"),
    ("SA-365", "Aerospatiale", "SA365", "helicopter", "Manual"),
    ("Aerospatiale SA", "Aerospatiale", "SA", "helicopter", "Manual"),
    ("Agusta A109", "Agusta", "A109", "helicopter", "Manual"),
    ("A109E", "Agusta", "A109E", "helicopter", "Manual"),

    # =========================================
    # PIPER
    # =========================================
    ("Piper PA-23", "Piper", "PA-23 Aztec", "multi-piston", "Manual"),
    ("Piper PA-24", "Piper", "PA-24 Comanche", "single-piston", "Manual"),
    ("PA-24-250", "Piper", "PA-24-250", "single-piston", "Manual"),
    ("Piper PA-28", "Piper", "PA-28 Cherokee", "single-piston", "Manual"),
    ("Piper PA-30", "Piper", "PA-30 Twin Comanche", "multi-piston", "Manual"),
    ("Piper PA-31", "Piper", "PA-31 Navajo", "multi-piston", "Manual"),
    ("PA-31-350", "Piper", "PA-31-350 Chieftain", "multi-piston", "Manual"),
    ("PA-31-310", "Piper", "PA-31-310", "multi-piston", "Manual"),
    ("Piper PA-32", "Piper", "PA-32 Cherokee Six", "single-piston", "Manual"),
    ("Piper PA-34", "Piper", "PA-34 Seneca", "multi-piston", "Manual"),
    ("PA23-250", "Piper", "PA-23-250 Aztec", "multi-piston", "Manual"),

    # =========================================
    # AERO COMMANDER / JET COMMANDER
    # =========================================
    ("Aero Commander", "Aero Commander", None, "multi-piston", "Manual"),
    ("Aero Commander 560", "Aero Commander", "560", "multi-piston", "Manual"),
    ("Aero Commander 560E", "Aero Commander", "560E", "multi-piston", "Manual"),
    ("Aero Commander 680", "Aero Commander", "680", "multi-piston", "Manual"),
    ("Aero Commander 1121", "Aero Commander", "1121 Jet Commander", "jet-regional", "Manual"),
    ("Commander 1121", "Aero Commander", "1121 Jet Commander", "jet-regional", "Manual"),
    ("Jet Commander", "Aero Commander", "Jet Commander", "jet-regional", "Manual"),
    ("Turbo Commander", "Aero Commander", "Turbo Commander", "turboprop", "Manual"),

    # =========================================
    # MARTIN
    # =========================================
    ("Martin 202", "Martin", "202", "multi-piston", "Manual"),
    ("Martin 404", "Martin", "404", "multi-piston", "Manual"),

    # =========================================
    # NORTH AMERICAN
    # =========================================
    ("North American TB-25", "North American", "TB-25", "multi-piston", "Manual"),
    ("North American SNJ", "North American", "SNJ", "single-piston", "Manual"),
    ("SNJ-4", "North American", "SNJ-4", "single-piston", "Manual"),
    ("SNJ-4N", "North American", "SNJ-4N", "single-piston", "Manual"),

    # =========================================
    # MILITARY / MISCELLANEOUS
    # =========================================
    ("F-4C", "McDonnell Douglas", "F-4C Phantom", "jet-regional", "Manual"),
    ("F-111", "General Dynamics", "F-111", "jet-regional", "Manual"),
    ("F111", "General Dynamics", "F-111", "jet-regional", "Manual"),
    ("F-106", "Convair", "F-106", "jet-regional", "Manual"),

    # =========================================
    # CONVAIR
    # =========================================
    ("Convair 240", "Convair", "240", "multi-piston", "Manual"),
    ("Convair 340", "Convair", "340", "multi-piston", "Manual"),
    ("Convair 440", "Convair", "440", "multi-piston", "Manual"),
    ("Convair 580", "Convair", "580", "turboprop", "Manual"),
    ("Convair 600", "Convair", "600", "turboprop", "Manual"),
    ("Convair 640", "Convair", "640", "turboprop", "Manual"),
    ("Convair 880", "Convair", "880", "jet-narrow", "Manual"),
    ("Convair 990", "Convair", "990", "jet-narrow", "Manual"),

    # =========================================
    # ATR / AVIONS DE TRANSPORT REGIONAL
    # =========================================
    ("ATR 42", "ATR", "42", "turboprop", "Manual"),
    ("ATR 72", "ATR", "72", "turboprop", "Manual"),
    ("ATR-42", "ATR", "42", "turboprop", "Manual"),
    ("ATR-72", "ATR", "72", "turboprop", "Manual"),
    ("ATR 72-212", "ATR", "72-212", "turboprop", "Manual"),
    ("Avions de Transport Regional", "ATR", None, "turboprop", "Manual"),

    # =========================================
    # BOEING / MCDONNELL DOUGLAS JETS
    # =========================================
    ("Boeing 707", "Boeing", "707", "jet-narrow", "Manual"),
    ("Boeing 720", "Boeing", "720", "jet-narrow", "Manual"),
    ("Boeing 727", "Boeing", "727", "jet-narrow", "Manual"),
    ("Boeing 727-100", "Boeing", "727-100", "jet-narrow", "Manual"),
    ("Boeing 727-200", "Boeing", "727-200", "jet-narrow", "Manual"),
    ("Boeing 737", "Boeing", "737", "jet-narrow", "Manual"),
    ("Boeing 737-200", "Boeing", "737-200", "jet-narrow", "Manual"),
    ("Boeing 737-300", "Boeing", "737-300", "jet-narrow", "Manual"),
    ("Boeing 747", "Boeing", "747", "jet-wide", "Manual"),
    ("Boeing 747-100", "Boeing", "747-100", "jet-wide", "Manual"),
    ("Boeing 747-200", "Boeing", "747-200", "jet-wide", "Manual"),
    ("Boeing 757", "Boeing", "757", "jet-narrow", "Manual"),
    ("Boeing 757-200", "Boeing", "757-200", "jet-narrow", "Manual"),
    ("Boeing 767", "Boeing", "767", "jet-wide", "Manual"),
    ("Boeing 777", "Boeing", "777", "jet-wide", "Manual"),
    ("Boeing MD-10", "Boeing", "MD-10", "jet-wide", "Manual"),
    ("Boeing MD-11", "Boeing", "MD-11", "jet-wide", "Manual"),
    ("Boeing MD-80", "Boeing", "MD-80", "jet-narrow", "Manual"),
    ("Boeing MD-83", "Boeing", "MD-83", "jet-narrow", "Manual"),
    ("MD-10-10F", "McDonnell Douglas", "MD-10-10F", "jet-wide", "Manual"),

    # =========================================
    # BRITISH AEROSPACE / BAe / JETSTREAM
    # =========================================
    ("British Aerospace", "British Aerospace", None, "turboprop", "Manual"),
    ("BAe", "British Aerospace", None, "turboprop", "Manual"),
    ("BAe 125", "British Aerospace", "125", "jet-regional", "Manual"),
    ("BAe 146", "British Aerospace", "146", "jet-regional", "Manual"),
    ("BAe J31", "British Aerospace", "Jetstream 31", "turboprop", "Manual"),
    ("BAe J32", "British Aerospace", "Jetstream 32", "turboprop", "Manual"),
    ("BAE-J3201", "British Aerospace", "Jetstream 32", "turboprop", "Manual"),
    ("J-3101", "British Aerospace", "Jetstream 31", "turboprop", "Manual"),
    ("J-3201", "British Aerospace", "Jetstream 32", "turboprop", "Manual"),
    ("Jetstream 31", "British Aerospace", "Jetstream 31", "turboprop", "Manual"),
    ("Jetstream 32", "British Aerospace", "Jetstream 32", "turboprop", "Manual"),
    ("BAC 1-11", "BAC", "1-11", "jet-narrow", "Manual"),
    ("BAC 111", "BAC", "1-11", "jet-narrow", "Manual"),
    ("BAC One-Eleven", "BAC", "1-11", "jet-narrow", "Manual"),
    ("Handley Page", "Handley Page", "HP-137", "turboprop", "Manual"),
    ("HP-137", "Handley Page", "HP-137", "turboprop", "Manual"),

    # =========================================
    # EMBRAER
    # =========================================
    ("Embraer EMB-110", "Embraer", "EMB-110", "turboprop", "Manual"),
    ("Embraer EMB-120", "Embraer", "EMB-120", "turboprop", "Manual"),
    ("Embraer EMB-500", "Embraer", "EMB-500 Phenom 100", "jet-regional", "Manual"),
    ("EMB-500", "Embraer", "EMB-500 Phenom 100", "jet-regional", "Manual"),
    ("Embraer ERJ-170", "Embraer", "ERJ-170", "jet-regional", "Manual"),
    ("ERJ-170", "Embraer", "ERJ-170", "jet-regional", "Manual"),
    ("ERJ-145", "Embraer", "ERJ-145", "jet-regional", "Manual"),
    ("E-170", "Embraer", "E-170", "jet-regional", "Manual"),
    ("E-175", "Embraer", "E-175", "jet-regional", "Manual"),
    ("E-190", "Embraer", "E-190", "jet-regional", "Manual"),
    ("E170", "Embraer", "E-170", "jet-regional", "Manual"),

    # =========================================
    # FOKKER
    # =========================================
    ("Fokker F27", "Fokker", "F27", "turboprop", "Manual"),
    ("Fokker F-27", "Fokker", "F27", "turboprop", "Manual"),
    ("Fokker F27-100", "Fokker", "F27-100", "turboprop", "Manual"),
    ("Fokker F28", "Fokker", "F28", "jet-regional", "Manual"),
    ("Fokker F-28", "Fokker", "F28", "jet-regional", "Manual"),
    ("Fokker 50", "Fokker", "50", "turboprop", "Manual"),
    ("Fokker 70", "Fokker", "70", "jet-regional", "Manual"),
    ("Fokker 100", "Fokker", "100", "jet-regional", "Manual"),
    ("F-28", "Fokker", "F28", "jet-regional", "Manual"),

    # =========================================
    # SAAB
    # =========================================
    ("SAAB 340", "SAAB", "340", "turboprop", "Manual"),
    ("SAAB 340B", "SAAB", "340B", "turboprop", "Manual"),
    ("Saab 340", "SAAB", "340", "turboprop", "Manual"),
    ("SAAB 2000", "SAAB", "2000", "turboprop", "Manual"),

    # =========================================
    # NORD
    # =========================================
    ("Nord 262", "Nord", "262", "turboprop", "Manual"),
    ("Nord Aviation", "Nord", "262", "turboprop", "Manual"),

    # =========================================
    # CASA
    # =========================================
    ("CASA C-212", "CASA", "C-212", "turboprop", "Manual"),
    ("CASA 212", "CASA", "C-212", "turboprop", "Manual"),
    ("C-212", "CASA", "C-212", "turboprop", "Manual"),
    ("C-212-CC", "CASA", "C-212-CC", "turboprop", "Manual"),
    ("Construcciones Aeronauticas", "CASA", "C-212", "turboprop", "Manual"),

    # =========================================
    # CIRRUS
    # =========================================
    ("Cirrus SR20", "Cirrus", "SR20", "single-piston", "Manual"),
    ("Cirrus SR22", "Cirrus", "SR22", "single-piston", "Manual"),
    ("Cirrus SR22T", "Cirrus", "SR22T", "single-piston", "Manual"),
    ("SR-20", "Cirrus", "SR20", "single-piston", "Manual"),
    ("SR-22", "Cirrus", "SR22", "single-piston", "Manual"),
    ("SR20", "Cirrus", "SR20", "single-piston", "Manual"),
    ("SR22", "Cirrus", "SR22", "single-piston", "Manual"),

    # =========================================
    # PILATUS
    # =========================================
    ("Pilatus PC-12", "Pilatus", "PC-12", "turboprop", "Manual"),
    ("PC-12", "Pilatus", "PC-12", "turboprop", "Manual"),
    ("PC12", "Pilatus", "PC-12", "turboprop", "Manual"),

    # =========================================
    # BALLOON
    # =========================================
    ("Hot Air Balloon", "Balloon", "Hot Air Balloon", "balloon", "Manual"),
    ("Balloon", "Balloon", "Balloon", "balloon", "Manual"),
    ("Balony Kubicek", "Balony Kubicek", "BB85Z", "balloon", "Manual"),
]


def insert_additional_patterns(conn, verbose=True):
    """Insert additional patterns into aircraft_lookup table."""
    cursor = conn.cursor()
    inserted = 0

    for pattern, make, model, category, source in ADDITIONAL_PATTERNS:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO aircraft_lookup
                (pattern, pattern_type, make, model, category, source, notes)
                VALUES (?, 'contains', ?, ?, ?, ?, 'Additional pattern')
            """, (pattern, make, model, category, source))
            if cursor.rowcount > 0:
                inserted += 1
        except Exception as e:
            if verbose:
                print(f"  Error: {pattern}: {e}")

    conn.commit()

    if verbose:
        print(f"Inserted {inserted} additional patterns")

    return inserted


if __name__ == "__main__":
    import sqlite3
    conn = sqlite3.connect("sqlite/riskradar.db")
    insert_additional_patterns(conn)
    conn.close()
