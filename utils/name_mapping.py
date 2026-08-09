COUNTRIES = ["Sudan"]

SUDAN_STATE_MAPPING = {
    "Al Gezira": "Al Jazirah",
    "Nile": "River Nile",
    "Eastern Darfur": "East Darfur",
    "Abyei PCA": "Abyei",
    "Aj Jazirah": "Al Jazirah",
}

SUDAN_PCODE_MAPPING = {
    "SD19": "Abyei",
    "SS80": "Abyei",  # South Sudan
    "SD15": "Aj Jazirah",
    "SD08": "Blue Nile",
    "SD06": "Central Darfur",
    "SD05": "East Darfur",
    "SD12": "Gedaref",
    "SD11": "Kassala",
    "SD01": "Khartoum",
    "SD02": "North Darfur",
    "SD13": "North Kordofan",
    "SD17": "Northern",
    "SD10": "Red Sea",
    "SD16": "River Nile",
    "SD14": "Sennar",
    "SD03": "South Darfur",
    "SD07": "South Kordofan",
    "SD04": "West Darfur",
    "SD18": "West Kordofan",
    "SD09": "White Nile",
}


def clean_state_names(value):
    return SUDAN_STATE_MAPPING.get(value, value)
