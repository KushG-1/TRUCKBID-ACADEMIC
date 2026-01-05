# Updated section in data_generator.py
"""
Complete Kathmandu Truck Data Generator with REAL prices
"""
import pandas as pd
import numpy as np
import random
from datetime import datetime

# REALISTIC KATHMANDU PRICING CONSTANTS
KTM_RATES = {
    "SMALL": {"rate": 18, "min": 400, "max": 1500},
    "MEDIUM": {"rate": 25, "min": 700, "max": 2500},  # Your 1,250 benchmark fits here
    "LARGE": {"rate": 35, "min": 1200, "max": 4000}
}

KATHMANDU_AREAS = [
    "Baneshwor", "Koteshwor", "New Road", "Patan", "Bhaktapur",
    "Kalanki", "Swayambhu", "Budhanilkantha", "Kirtipur", "Gaushala",
    "Chabahil", "Thamel", "Maharajgunj", "Lazimpat", "Dillibazar"
]

def calculate_base_price(distance_km, truck_category):
    """Calculate realistic base price for Kathmandu"""
    rate = KTM_RATES[truck_category]["rate"]
    min_charge = KTM_RATES[truck_category]["min"]
    
    base_price = distance_km * rate
    
    # Apply minimum charge
    if base_price < min_charge:
        base_price = min_charge
    
    return base_price

def apply_kathmandu_factors(base_price, factors):
    """
    Apply Kathmandu-specific pricing factors
    factors dict should contain: traffic_level, is_peak_hour, time_of_day, distance_km
    """
    adjusted = base_price
    
    # Traffic adjustments (Kathmandu specific)
    traffic_multipliers = {
        "Light": 1.0,       # Normal
        "Medium": 1.1,      # 10% more
        "Heavy": 1.3,       # 30% more (common in Kathmandu)
        "Very Heavy": 1.5   # 50% more (ring road, peak hours)
    }
    
    # Apply traffic multiplier
    traffic = factors.get('traffic_level', 'Medium')
    adjusted *= traffic_multipliers.get(traffic, 1.1)
    
    # Peak hour adjustment (7-9 AM, 5-7 PM)
    if factors.get('is_peak_hour', 0):
        adjusted *= 1.2  # 20% more during peak
    
    # Time of day adjustments
    time_of_day = factors.get('time_of_day', 'Afternoon')
    if time_of_day == "Night":
        adjusted *= 0.9  # 10% discount at night
    
    # Distance-based adjustment
    distance = factors.get('distance_km', 0)
    if distance > 15:  # Long trips in Kathmandu
        adjusted *= 0.95  # 5% discount
    
    # Add small randomness (±10%) - real market variation
    adjusted *= random.uniform(0.9, 1.1)
    
    # Apply maximum cap
    truck_category = factors.get('truck_category', 'MEDIUM')
    max_price = KTM_RATES[truck_category]["max"]
    if adjusted > max_price:
        adjusted = max_price
    
    return round(adjusted, 2)

def get_realistic_traffic(pickup, delivery, time_of_day, is_peak):
    """Get realistic traffic for Kathmandu routes"""
    
    # Always congested routes in Kathmandu
    always_heavy = [
        ("Kalanki", "Koteshwor"),
        ("Baneshwor", "Koteshwor"),
        ("New Road", "Baneshwor"),
        ("Kalanki", "Swayambhu")
    ]
    
    # Check if this is a known congested route
    route = (pickup, delivery)
    reverse_route = (delivery, pickup)
    
    if route in always_heavy or reverse_route in always_heavy:
        return "Heavy" if random.random() > 0.3 else "Very Heavy"
    
    # Peak hour traffic
    if is_peak:
        return random.choice(["Heavy", "Very Heavy"])
    
    # Night time
    if time_of_day == "Night":
        return "Light"
    
    # Normal daytime
    return random.choice(["Light", "Medium"])

def get_time_of_day():
    """Random time of day with realistic distribution"""
    times = ["Morning", "Afternoon", "Evening", "Night"]
    weights = [0.3, 0.4, 0.2, 0.1]  # More trips in afternoon
    return random.choices(times, weights=weights)[0]

def get_realistic_distance(pickup, delivery):
    """
    Get realistic distance between Kathmandu areas
    Based on actual Kathmandu geography
    """
    # Realistic distances between Kathmandu areas
    distance_map = {
        ("Baneshwor", "Koteshwor"): 2.8,
        ("Baneshwor", "New Road"): 3.5,
        ("Baneshwor", "Patan"): 8.3,
        ("Koteshwor", "Kalanki"): 1.9,
        ("New Road", "Patan"): 4.8,
        ("New Road", "Bhaktapur"): 12.7,
        ("Patan", "Bhaktapur"): 8.9,
        ("Kalanki", "Swayambhu"): 5.8,
        ("Swayambhu", "Chabahil"): 2.9,
        ("Chabahil", "Gaushala"): 1.5,
        ("Patan", "Kirtipur"): 3.5,
        ("Baneshwor", "Chabahil"): 3.8,
        ("Thamel", "New Road"): 0.8,
        ("Maharajgunj", "Lazimpat"): 0.7
    }
    
    # Check direct mapping
    if (pickup, delivery) in distance_map:
        return distance_map[(pickup, delivery)]
    
    # Check reverse mapping
    if (delivery, pickup) in distance_map:
        return distance_map[(delivery, pickup)]
    
    # If not in map, calculate based on area clusters
    # Central areas are close, far areas are farther
    central_areas = ["New Road", "Baneshwor", "Thamel", "Dillibazar"]
    east_areas = ["Koteshwor", "Chabahil", "Gaushala"]
    west_areas = ["Kalanki", "Swayambhu"]
    south_areas = ["Patan", "Kirtipur"]
    north_areas = ["Bhaktapur", "Budhanilkantha"]
    
    # Determine cluster distance
    if (pickup in central_areas and delivery in central_areas):
        return round(random.uniform(1, 4), 1)
    elif (pickup in east_areas and delivery in west_areas):
        return round(random.uniform(8, 15), 1)
    elif (pickup in south_areas and delivery in north_areas):
        return round(random.uniform(10, 18), 1)
    else:
        return round(random.uniform(5, 12), 1)

def generate_kathmandu_data(num_samples=500):
    """Generate Kathmandu dataset with REAL prices"""
    
    print("🚛 Generating Kathmandu truck dataset...")
    
    data = []
    
    for i in range(num_samples):
        # Pick random Kathmandu areas (different pickup/delivery)
        pickup = random.choice(KATHMANDU_AREAS)
        delivery = random.choice([a for a in KATHMANDU_AREAS if a != pickup])
        
        # Get REALISTIC distance
        distance = get_realistic_distance(pickup, delivery)
        
        # Ensure realistic Kathmandu distance (1-20km)
        if distance < 1:
            distance = 1.5
        if distance > 20:
            distance = 18.5
        
        # Truck distribution (Kathmandu reality: more small vehicles)
        rand = random.random()
        if rand < 0.6:  # 60% small trucks (tempos/vans)
            category = "SMALL"
        elif rand < 0.9:  # 30% medium trucks
            category = "MEDIUM"
        else:  # 10% large trucks
            category = "LARGE"
        
        # Time factors
        time_of_day = get_time_of_day()
        is_peak = 1 if time_of_day in ["Morning", "Evening"] else 0
        
        # Traffic based on route and time
        traffic = get_realistic_traffic(pickup, delivery, time_of_day, is_peak)
        
        # Calculate base price
        base_price = calculate_base_price(distance, category)
        
        # Apply Kathmandu factors - FIXED FUNCTION NAME
        factors = {
            'traffic_level': traffic,
            'is_peak_hour': is_peak,
            'time_of_day': time_of_day,
            'distance_km': distance,
            'truck_category': category
        }
        final_price = apply_kathmandu_factors(base_price, factors)
        
        # Ensure final price is realistic for Kathmandu
        if final_price < KTM_RATES[category]["min"]:
            final_price = KTM_RATES[category]["min"]
        if final_price > KTM_RATES[category]["max"]:
            final_price = KTM_RATES[category]["max"]
        
        # Create record
        record = {
            'id': i + 1,
            'truck_category': category,
            'distance_km': distance,
            'traffic_level': traffic,
            'time_of_day': time_of_day,
            'is_peak_hour': is_peak,
            'pickup_area': pickup,
            'delivery_area': delivery,
            'base_price_npr': round(base_price, 2),
            'accepted_price_npr': round(final_price, 2),
            'price_per_km': round(final_price / distance, 2) if distance > 0 else 0
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    filename = 'data.csv'
    df.to_csv(filename, index=False)
    
    print_summary(df)
    
    return df

def print_summary(df):
    """Print dataset summary"""
    print("\n" + "="*60)
    print("📊 KATHMANDU TRUCK DATASET SUMMARY")
    print("="*60)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Records: {len(df)}")
    print(f"  Avg Distance: {df['distance_km'].mean():.1f} km")
    print(f"  Avg Price: NPR {df['accepted_price_npr'].mean():,.0f}")
    print(f"  Price Range: NPR {df['accepted_price_npr'].min():,.0f} - NPR {df['accepted_price_npr'].max():,.0f}")
    
    print(f"\n🚛 Truck Category Distribution:")
    for category in ["SMALL", "MEDIUM", "LARGE"]:
        cat_data = df[df['truck_category'] == category]
        count = len(cat_data)
        avg_price = cat_data['accepted_price_npr'].mean()
        avg_rate = cat_data['price_per_km'].mean()
        print(f"  {category}: {count:3d} trips | Avg: NPR {avg_price:,.0f} | Rate: NPR {avg_rate:.1f}/km")
    
    print(f"\n📍 Most Common Routes:")
    route_counts = df.groupby(['pickup_area', 'delivery_area']).size()
    top_routes = route_counts.sort_values(ascending=False).head(5)
    
    for (pickup, delivery), count in top_routes.items():
        route_data = df[(df['pickup_area']==pickup) & (df['delivery_area']==delivery)]
        avg_price = route_data['accepted_price_npr'].mean()
        avg_dist = route_data['distance_km'].mean()
        print(f"  {pickup:15} → {delivery:15}: {count:2d} trips | {avg_dist:.1f}km | NPR {avg_price:,.0f}")
    
    print(f"\n⏰ Time Analysis:")
    # Peak vs Off-peak
    peak_avg = df[df['is_peak_hour'] == 1]['accepted_price_npr'].mean()
    offpeak_avg = df[df['is_peak_hour'] == 0]['accepted_price_npr'].mean()
    print(f"  Peak hours:    NPR {peak_avg:,.0f}")
    print(f"  Off-peak:      NPR {offpeak_avg:,.0f}")
    print(f"  Peak premium:  {((peak_avg-offpeak_avg)/offpeak_avg*100):.1f}%")
    
    print(f"\n🚦 Traffic Impact:")
    for traffic in ["Light", "Medium", "Heavy", "Very Heavy"]:
        traffic_data = df[df['traffic_level'] == traffic]
        if len(traffic_data) > 0:
            avg_price = traffic_data['accepted_price_npr'].mean()
            print(f"  {traffic:12}: {len(traffic_data):3d} trips | NPR {avg_price:,.0f}")
    
    print(f"\n💾 Saved to: kathmandu_truck_data_{len(df)}.csv")
    print("="*60)

def validate_prices(df):
    """Validate that prices match Kathmandu reality"""
    print("\n🔍 PRICE VALIDATION")
    print("-"*40)
    
    # Check medium truck average (should be ~1,250)
    medium_data = df[df['truck_category'] == 'MEDIUM']
    medium_avg = medium_data['accepted_price_npr'].mean()
    
    print(f"Medium Truck Average: NPR {medium_avg:,.0f}")
    
    if 1000 <= medium_avg <= 1500:
        print("✅ PERFECT! Matches Kathmandu reality (~NPR 1,250)")
    elif 800 <= medium_avg <= 1800:
        print("⚠️  Acceptable range (NPR 800-1,800)")
    else:
        print(f"❌ Needs adjustment (should be ~NPR 1,250)")
    
    # Check price distribution
    print(f"\n💰 Price Distribution:")
    price_ranges = [
        ("Under 1,000", df[df['accepted_price_npr'] < 1000]),
        ("1,000-1,500", df[(df['accepted_price_npr'] >= 1000) & (df['accepted_price_npr'] < 1500)]),
        ("1,500-2,000", df[(df['accepted_price_npr'] >= 1500) & (df['accepted_price_npr'] < 2000)]),
        ("Over 2,000", df[df['accepted_price_npr'] >= 2000])
    ]
    
    for label, data_range in price_ranges:
        count = len(data_range)
        percent = (count / len(df)) * 100
        print(f"  {label:15}: {count:3d} trips ({percent:.1f}%)")
    
    return True

if __name__ == "__main__":
    # Generate dataset
    df = generate_kathmandu_data(500)
    
    # Validate prices
    validate_prices(df)
    
    # Show sample data
    print("\n📄 SAMPLE DATA (first 5 rows):")
    print(df[['pickup_area', 'delivery_area', 'distance_km', 
              'truck_category', 'traffic_level', 'accepted_price_npr']].head())
              