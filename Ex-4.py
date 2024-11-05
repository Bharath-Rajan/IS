Ex-4

1) Implement a Crop Yield Estimator that calculates the expected yield of various crops based on factors such as soil quality, fertilizer use, and weather conditions. Use the estimator to analyze potential yields for different crop types over a specified growing season. 

import numpy as np

class AquacultureProfitEstimator:
    def __init__(self, initial_investment, operational_costs, selling_price_per_kg, production_rate_per_ha):
        self.initial_investment = initial_investment  # Initial investment in aquaculture setup
        self.operational_costs = operational_costs      # Operational costs per year
        self.selling_price_per_kg = selling_price_per_kg  # Selling price per kg of fish
        self.production_rate_per_ha = production_rate_per_ha  # Production rate in kg per hectare

    def estimate_profit(self, area_ha, years):
        total_revenue = area_ha * self.production_rate_per_ha * self.selling_price_per_kg * years
        total_costs = self.initial_investment + (self.operational_costs * years)
        profit = total_revenue - total_costs
        return profit

    def decision_analysis(self, area_options, years):
       results = {}
        for area in area_options:
            profit = self.estimate_profit(area, years)
            results[area] = profit
        return results

# Example usage
if __name__ == "__main__":
    # Parameters
    initial_investment = 50000  # Initial investment in dollars
    operational_costs = 10000    # Annual operational costs in dollars
    selling_price_per_kg = 5     # Selling price per kg of fish in dollars
    production_rate_per_ha = 1000 # Production rate in kg/ha/year

    # Create an estimator
    estimator = AquacultureProfitEstimator(initial_investment, operational_costs, selling_price_per_kg, production_rate_per_ha)

    # Define area options and years for analysis
    area_options = [1, 5, 10]  # Areas in hectares
    years = 5                   # Duration in years

    # Perform decision analysis
    profit_analysis = estimator.decision_analysis(area_options, years)

    # Print results
    for area, profit in profit_analysis.items():
        print(f"Profit for {area} ha over {years} years: ${profit:.2f}") 
2) Develop a simulation to estimate the annual profit of different crop options (e.g., Corn and Wheat) based on varying weather conditions and soil moisture levels over a year (365 days). The estimator should consider the expected yield influenced by weather (Sunny, Rainy, Cloudy) and soil moisture, with random variations added to reflect real-world uncertainty.
import random
num_days = 365
weather_mapping = {
'Sunny': 0.2,
'Rainy': 0.2,
'Cloudy': 0.4
}
weather_data = [random.choice(['Sunny', 'Rainy', 'Cloudy']) for _ in range(num_days)]
soil_moisture_data = [random.uniform(0.1, 0.9) for _ in range(num_days)]
crop_yield_data = [500 + (weather_mapping[w] * 50) - (m * 100) + random.randint(-20, 20) for w, m in zip(weather_data, soil_moisture_data)]
crop_options = {'Corn': {'yield': 500, 'profit_per_acre': 300},'Wheat': {'yield': 300, 'profit_per_acre': 200} }
annual_profit_corn = 0
annual_profit_wheat = 0
for day in range(num_days):
    weather = weather_data[day]
    soil_moisture = soil_moisture_data[day]
    expected_yield_corn = crop_options['Corn']['yield']
    expected_yield_wheat = crop_options['Wheat']['yield']
    daily_profit_corn = crop_options['Corn']['profit_per_acre'] * expected_yield_corn
    daily_profit_wheat = crop_options['Wheat']['profit_per_acre'] * expected_yield_wheat
    if daily_profit_corn > daily_profit_wheat:
        annual_profit_corn += daily_profit_corn
    else:
        annual_profit_wheat += daily_profit_wheat
print("Annual profit for Corn:", annual_profit_corn)
print("Annual profit for Wheat:", annual_profit_wheat)
3) Design a program to evaluate the profitability of a livestock farming venture by estimating the annual profit based on various parameters, including initial investment, operational costs, selling price per animal, and production rate per animal. 

class LivestockProfitEstimator:
    def __init__(self, initial_investment, operational_costs, selling_price_per_animal, production_rate_per_animal):
        self.initial_investment = initial_investment  # Initial investment in livestock setup
        self.operational_costs = operational_costs      # Operational costs per year
        self.selling_price_per_animal = selling_price_per_animal  # Selling price per animal
        self.production_rate_per_animal = production_rate_per_animal  # Production rate per animal

    def estimate_profit(self, num_animals, years):
        total_revenue = num_animals * self.production_rate_per_animal * self.selling_price_per_animal * years
        total_costs = self.initial_investment + (self.operational_costs * years)
        profit = total_revenue - total_costs
        return profit

    def decision_analysis(self, animal_options, years):
        results = {}
        for num_animals in animal_options:
            profit = self.estimate_profit(num_animals, years)
            results[num_animals] = profit
        return results

# Example usage
if __name__ == "__main__":
    # Parameters
    initial_investment = 30000  # Initial investment in dollars
    operational_costs = 5000     # Annual operational costs in dollars
    selling_price_per_animal = 1500  # Selling price per animal in dollars
    production_rate_per_animal = 1    # Production rate per animal (e.g., number of offspring)

    # Create an estimator
    estimator = LivestockProfitEstimator(initial_investment, operational_costs, selling_price_per_animal, production_rate_per_animal)

    # Define animal options and years for analysis
    animal_options = [10, 50, 100]  # Number of animals
    years = 5                        # Duration in years

    # Perform decision analysis
    profit_analysis = estimator.decision_analysis(animal_options, years)

    # Print results
    for num_animals, profit in profit_analysis.items():
        print(f"Profit for {num_animals} animals over {years} years: ${profit:.2f}") 
4) Design a program to evaluate the profitability of a gardening venture by estimating the annual profit based on various parameters such as initial investment, operational costs, selling price per plant, and production rate per bed.

class GardenProfitEstimator:
    def __init__(self, initial_investment, operational_costs, selling_price_per_plant, production_rate_per_bed):
        self.initial_investment = initial_investment  # Initial investment in garden setup
        self.operational_costs = operational_costs      # Operational costs per year
        self.selling_price_per_plant = selling_price_per_plant  # Selling price per plant
        self.production_rate_per_bed = production_rate_per_bed  # Production rate per bed

    def estimate_profit(self, num_beds, years):
        total_revenue = num_beds * self.production_rate_per_bed * self.selling_price_per_plant * years
        total_costs = self.initial_investment + (self.operational_costs * years)
        profit = total_revenue - total_costs
        return profit

    def decision_analysis(self, bed_options, years):
        results = {}
        for num_beds in bed_options:
            profit = self.estimate_profit(num_beds, years)
            results[num_beds] = profit
        return results

# Example usage
if __name__ == "__main__":
    # Parameters
    initial_investment = 10000  # Initial investment in dollars
    operational_costs = 2000     # Annual operational costs in dollars
    selling_price_per_plant = 5   # Selling price per plant in dollars
    production_rate_per_bed = 100 # Production rate per bed (e.g., number of plants)

    # Create an estimator
    estimator = GardenProfitEstimator(initial_investment, operational_costs, selling_price_per_plant, production_rate_per_bed)

    # Define bed options and years for analysis
    bed_options = [5, 10, 20]  # Number of beds
    years = 5                   # Duration in years

    # Perform decision analysis
    profit_analysis = estimator.decision_analysis(bed_options, years)

    # Print results
    for num_beds, profit in profit_analysis.items():
        print(f"Profit for {num_beds} beds over {years} years: ${profit:.2f}")
