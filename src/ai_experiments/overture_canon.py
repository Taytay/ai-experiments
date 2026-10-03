"""Overture places' basic_category (284 values in the US, release 2026-09-23.1) -> the canonical kinds of `ai_experiments.canon`
(CATEGORY_OPTIONS: 41 kinds of spending; PLAN step 173, owner 2026-10-03: "Our canonical 12 should be broadened to fit a larger set").
Written by hand from the category names; None = a place nobody pays as a merchant (nature, monuments, public safety) or one with no
single kind. Reviewed against the owner's budget only as aggregates (REPORT 161).
"""
from ai_experiments import canon as C

GROC, REST, FAST, COFFEE, BAR = "grocery store", "restaurant", "fast food", "coffee shop or bakery", "bar or liquor store"
GAS, CONV, AUTO, PARK, TRANSIT, TRAVEL = "gas station", "convenience store", "car repair or parts", "parking or tolls", "public transit or rideshare", "airline, hotel or car rental"
UTIL, PHONE, RENT, INS = "utility (electric, gas, water, trash)", "phone or internet provider", "rent, mortgage or HOA", "insurance"
DOC, PHARM, GYM, SALON = "doctor, dentist, hospital or optician", "pharmacy", "gym or fitness", "salon, barber or personal care"
ENT, HOBBY, BIGBOX, CLOTH = "entertainment (movies, events, games)", "books, hobby or sports store", "big-box or online store (sells many kinds of things)", "clothing or department store"
HOME, ELEC, PET, KIDS, EDU = "home improvement, furniture or household", "electronics store", "pet store or vet", "childcare, school or kids' activities", "education or tuition"
GIFT, CHARITY, BANK, GOV, BIZ, LAUNDRY, EVENT = ("gifts or florist", "charity or church", "bank fee, cash withdrawal or card payment", "taxes or government",
                                                 "business services or supplies", "laundry or dry cleaning", "wedding or event vendor")

MAP = {
    # food and drink
    "restaurant": REST, "casual_eatery": REST, "food_truck_stand": REST, "food_court": REST, "food_and_drink": REST, "food_service": REST,
    "fast_food_restaurant": FAST,
    "coffee_shop": COFFEE, "cafe": COFFEE, "smoothie_juice_bar": COFFEE, "non_alcoholic_beverage_venue": COFFEE,
    "bar": BAR, "lounge": BAR, "brewery": BAR, "winery": BAR, "distillery": BAR, "alcoholic_beverage_venue": BAR, "dance_club": BAR, "nightlife_venue": BAR,
    "food_and_beverage_store": GROC, "farmers_market": GROC, "market": GROC, "farm": GROC, "superstore": GROC,
    "convenience_store": CONV,
    # car and transport
    "gas_station": GAS, "fueling_station": GAS, "ev_charging_station": GAS,
    "automotive_service": AUTO, "vehicle_parts_store": AUTO, "vehicle_service": AUTO, "auto_dealer": AUTO, "vehicle_dealer": AUTO,
    "parking": PARK, "parking_garage": PARK, "parking_lot": PARK, "toll_station": PARK, "park_and_ride": PARK,
    "taxi_or_ride_share_service": TRANSIT, "public_transit_facility_or_service": TRANSIT, "train_station": TRANSIT, "ferry_service": TRANSIT,
    "ground_transport_facility_or_service": TRANSIT, "rail_facility_or_service": TRANSIT,
    "hotel": TRAVEL, "lodging": TRAVEL, "resort": TRAVEL, "bed_and_breakfast": TRAVEL, "inn": TRAVEL, "private_lodging": TRAVEL, "campground": TRAVEL,
    "rv_park": TRAVEL, "airport": TRAVEL, "air_transport_facility_or_service": TRAVEL, "travel_service": TRAVEL, "travel_and_transportation": TRAVEL,
    "rental_service": TRAVEL,
    # home and bills
    "public_utility": UTIL, "water_utility_provider": UTIL, "electric_utility_provider": UTIL, "natural_gas_utility_provider": UTIL,
    "b2b_energy_and_utility_service": UTIL, "utility_energy_infrastructure": UTIL,
    "telecommunications_service": PHONE,
    "real_estate_service": RENT, "apartment": RENT, "condominium": RENT, "housing_or_property_service": RENT, "senior_living_facility": RENT,
    "hardware_home_and_garden_store": HOME, "home_service": HOME, "building_or_construction_service": HOME, "storage_facility": HOME,
    "design_service": HOME, "security_service": HOME, "environmental_or_ecological_service": HOME, "agricultural_service": HOME,
    # health and personal
    "health_care": DOC, "specialized_health_care": DOC, "dental_clinic": DOC, "diagnostics_imaging_or_lab_service": DOC,
    "behavioral_or_mental_health_clinic": DOC, "complementary_and_alternative_medicine": DOC, "medical_service": DOC,
    "physical_medicine_and_rehabilitation": DOC, "primary_care_or_general_clinic": DOC, "outpatient_care_facility": DOC,
    "vision_or_eye_care_clinic": DOC, "surgery": DOC, "hospital": DOC, "reproductive_perinatal_and_womens_care": DOC,
    "specialized_medical_facility": DOC, "pediatric_clinic": DOC, "emergency_or_urgent_care_facility": DOC, "emergency_department": DOC,
    "urgent_care_clinic": DOC, "walk_in_clinic": DOC, "specialty_hospital": DOC, "general_hospital": DOC,
    "pharmacy_and_drug_store": PHARM,
    "gym": GYM, "fitness_studio": GYM, "sport_or_fitness_facility": GYM, "sport_or_recreation_club": GYM, "swimming_pool": GYM,
    "personal_or_beauty_service": SALON, "wellness_service": SALON, "personal_care_and_beauty_store": SALON,
    # shopping
    "fashion_and_apparel_store": CLOTH, "department_store": CLOTH, "shopping_mall": CLOTH,
    "discount_store": BIGBOX, "warehouse_club_store": BIGBOX, "shopping": BIGBOX, "second_hand_store": BIGBOX, "specialty_store": BIGBOX,
    "electronics_store": ELEC,
    "sporting_goods_store": HOBBY, "arts_crafts_and_hobby_store": HOBBY, "books_music_and_video_store": HOBBY, "toys_and_games_store": HOBBY,
    "musical_instrument_and_pro_audio_store": HOBBY, "recreational_equipment_rental": HOBBY,
    "flowers_and_gifts_store": GIFT,
    "animal_or_pet_service": PET, "animal_and_pet_store": PET,
    # entertainment
    "movie_theater": ENT, "music_venue": ENT, "event_venue": ENT, "stadium_arena": ENT, "amusement_park": ENT, "theatre_venue": ENT,
    "performing_arts_venue": ENT, "arcade": ENT, "casino": ENT, "gaming_venue": ENT, "museum": ENT, "art_gallery": ENT, "golf_course": ENT,
    "bowling_alley": ENT, "skating_rink": ENT, "comedy_club": ENT, "zoo": ENT, "aquarium": ENT, "amusement_attraction": ENT,
    "animal_attraction": ENT, "festival_venue": ENT, "arts_and_entertainment": ENT, "sports_and_recreation": ENT, "science_attraction": ENT,
    "planetarium": ENT, "fairgrounds": ENT, "rodeo": ENT, "ticket_office_or_booth": ENT, "sport_team": ENT, "country_club": ENT,
    "adult_entertainment_venue": ENT,
    # kids, school
    "elementary_school": KIDS, "preschool": KIDS, "middle_school": KIDS, "high_school": KIDS, "kindergarten": KIDS, "youth_organization": KIDS,
    "sport_league": KIDS, "family_service": KIDS, "tutoring_service": KIDS,
    "college_university": EDU, "place_of_learning": EDU, "specialty_school": EDU, "education": EDU, "educational_service": EDU,
    "educational_facility": EDU, "campus_building": EDU, "academy": EDU, "class_venue": EDU,
    # money, government, giving, business
    "financial_service": BANK, "bank_or_credit_union": BANK, "atm": BANK,
    "government_office": GOV, "community_and_government": GOV, "courthouse": GOV, "government_department": GOV, "school_district_office": GOV,
    "christian_place_of_worship": CHARITY, "religious_organization": CHARITY, "social_or_community_service": CHARITY, "civic_organization": CHARITY,
    "jewish_place_of_worship": CHARITY, "muslim_place_of_worship": CHARITY, "buddhist_place_of_worship": CHARITY, "hindu_place_of_worship": CHARITY,
    "place_of_worship": CHARITY, "food_bank": CHARITY, "labor_union": CHARITY, "political_organization": CHARITY,
    "professional_service": BIZ, "technical_service": BIZ, "attorney_or_law_firm": BIZ, "legal_service": BIZ, "corporate_or_business_office": BIZ,
    "manufacturer": BIZ, "media_service": BIZ, "supplier_or_distributor": BIZ, "b2b_service": BIZ, "b2b_transportation_and_storage_service": BIZ,
    "printing_service": BIZ, "shipping_or_delivery_service": BIZ, "b2b_office_and_professional_service": BIZ, "wholesaler": BIZ,
    "office_supply_store": BIZ, "b2b_industrial_and_machine_service": BIZ, "b2b_science_and_technology_service": BIZ, "radio_station": BIZ,
    "industrial_facility_or_service": BIZ, "research_institute": BIZ,
    "laundry_service": LAUNDRY,
    "event_or_party_service": EVENT,
}
# Everything else (parks, lakes, historic sites, monuments, public safety, landmarks, social clubs, psychics, cemeteries ...) has no kind.

assert all(v in C.CATEGORY_OPTIONS for v in MAP.values()), {v for v in MAP.values() if v not in C.CATEGORY_OPTIONS}


def kind(basic_category):
    return MAP.get(basic_category)
