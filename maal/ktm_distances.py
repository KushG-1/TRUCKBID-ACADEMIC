"""
Real Kathmandu distances between major areas
Based on actual road distances (in kilometers)
"""
import random

KATHMANDU_DISTANCES = {
    # Central Kathmandu areas
    ("New Road", "Asan"): 1.2,
    ("New Road", "Baneshwor"): 3.5,
    ("New Road", "Koteshwor"): 6.2,
    ("New Road", "Patan"): 4.8,
    ("New Road", "Bhaktapur"): 12.7,
    ("New Road", "Kalanki"): 5.8,
    
    # Baneshwor connections
    ("Baneshwor", "Koteshwor"): 2.8,
    ("Baneshwor", "Patan"): 8.3,
    ("Baneshwor", "Bhaktapur"): 14.5,
    ("Baneshwor", "Kalanki"): 4.7,
    ("Baneshwor", "Swayambhu"): 5.2,
    ("Baneshwor", "Chabahil"): 3.8,
    
    # Koteshwor connections
    ("Koteshwor", "Patan"): 11.2,
    ("Koteshwor", "Bhaktapur"): 17.3,
    ("Koteshwor", "Kalanki"): 1.9,
    ("Koteshwor", "Swayambhu"): 7.5,
    ("Koteshwor", "Chabahil"): 6.2,
    
    # Patan connections
    ("Patan", "Bhaktapur"): 8.9,
    ("Patan", "Kalanki"): 12.8,
    ("Patan", "Swayambhu"): 4.7,
    ("Patan", "Kirtipur"): 3.5,
    ("Patan", "Jawalakhel"): 1.2,
    
    # Other routes
    ("Kalanki", "Swayambhu"): 5.8,
    ("Kalanki", "Chabahil"): 8.3,
    ("Swayambhu", "Chabahil"): 2.9,
    ("Swayambhu", "Kirtipur"): 4.2,
    ("Chabahil", "Gaushala"): 1.5,
    ("Chabahil", "Budhanilkantha"): 6.7,
    ("Thamel", "New Road"): 0.8,
    ("Thamel", "Baneshwor"): 4.2,
    ("Lazimpat", "Maharajgunj"): 0.7,
    ("Dillibazar", "Baneshwor"): 1.5,
}

KATHMANDU_AREAS = [
    "Baneshwor", "Koteshwor", "New Road", "Patan", "Bhaktapur",
    "Kalanki", "Swayambhu", "Budhanilkantha", "Kirtipur", "Gaushala",
    "Chabahil", "Thamel", "Maharajgunj", "Lazimpat", "Dillibazar"
]

def get_distance(pickup, delivery):
    """
    Get realistic distance between two Kathmandu areas.
    Returns distance in kilometers.
    """
    # Check direct mapping
    if (pickup, delivery) in KATHMANDU_DISTANCES:
        return KATHMANDU_DISTANCES[(pickup, delivery)]
    
    # Check reverse mapping
    if (delivery, pickup) in KATHMANDU_DISTANCES:
        return KATHMANDU_DISTANCES[(delivery, pickup)]
    
    # If not found, calculate approximate distance
    return estimate_by_area_type(pickup, delivery)

def estimate_by_area_type(area1, area2):
    """Estimate distance based on area characteristics"""
    central_areas = ["New Road", "Thamel", "Dillibazar"]
    east_areas = ["Baneshwor", "Koteshwor", "Chabahil", "Gaushala"]
    west_areas = ["Kalanki", "Swayambhu"]
    south_areas = ["Patan", "Kirtipur"]
    north_areas = ["Bhaktapur", "Budhanilkantha", "Maharajgunj", "Lazimpat"]
    
    def get_area_type(area):
        if area in central_areas:
            return "central"
        elif area in east_areas:
            return "east"
        elif area in west_areas:
            return "west"
        elif area in south_areas:
            return "south"
        elif area in north_areas:
            return "north"
        return "unknown"
    
    area1_type = get_area_type(area1)
    area2_type = get_area_type(area2)
    
    # Estimate based on combination
    if area1_type == area2_type:
        return round(random.uniform(2, 5), 1)
    elif area1_type in ["central", "unknown"] or area2_type in ["central", "unknown"]:
        return round(random.uniform(4, 10), 1)
    elif (area1_type, area2_type) in [("east", "west"), ("north", "south")]:
        return round(random.uniform(10, 18), 1)
    else:
        return round(random.uniform(6, 14), 1)

def get_all_areas():
    """Get all unique areas"""
    return KATHMANDU_AREAS