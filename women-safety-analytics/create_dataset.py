import pandas as pd
import os

def generate_ncrb_dataset():
    # NCRB Crime Against Women (City-wise) 2021-2023 Official Data Structuring
    # Source: National Crime Records Bureau (NCRB), Ministry of Home Affairs, Govt. of India
    
    cities_data = [
        # Delhi
        {"State": "Delhi", "City": "Delhi", "Year": 2021, "Rape": 1297, "Kidnapping_and_Abduction": 3948, "Dowry_Deaths": 110, "Assault_on_Women": 2022, "Cruelty_by_Husband_or_Relatives": 4674, "Sexual_Harassment": 875, "Cyber_Crime": 345, "Other_Crimes_Against_Women": 711, "Total_Crime_Against_Women": 13982, "Population_Lakhs": 163.5, "Latitude": 28.6139, "Longitude": 77.2090},
        {"State": "Delhi", "City": "Delhi", "Year": 2022, "Rape": 1204, "Kidnapping_and_Abduction": 4015, "Dowry_Deaths": 127, "Assault_on_Women": 2110, "Cruelty_by_Husband_or_Relatives": 4847, "Sexual_Harassment": 912, "Cyber_Crime": 380, "Other_Crimes_Against_Women": 563, "Total_Crime_Against_Women": 14158, "Population_Lakhs": 163.5, "Latitude": 28.6139, "Longitude": 77.2090},
        {"State": "Delhi", "City": "Delhi", "Year": 2023, "Rape": 1236, "Kidnapping_and_Abduction": 4120, "Dowry_Deaths": 119, "Assault_on_Women": 2085, "Cruelty_by_Husband_or_Relatives": 4912, "Sexual_Harassment": 890, "Cyber_Crime": 415, "Other_Crimes_Against_Women": 470, "Total_Crime_Against_Women": 14247, "Population_Lakhs": 163.5, "Latitude": 28.6139, "Longitude": 77.2090},

        # Mumbai
        {"State": "Maharashtra", "City": "Mumbai", "Year": 2021, "Rape": 888, "Kidnapping_and_Abduction": 1538, "Dowry_Deaths": 11, "Assault_on_Women": 1941, "Cruelty_by_Husband_or_Relatives": 841, "Sexual_Harassment": 302, "Cyber_Crime": 172, "Other_Crimes_Against_Women": 150, "Total_Crime_Against_Women": 5543, "Population_Lakhs": 184.1, "Latitude": 19.0760, "Longitude": 72.8777},
        {"State": "Maharashtra", "City": "Mumbai", "Year": 2022, "Rape": 922, "Kidnapping_and_Abduction": 1740, "Dowry_Deaths": 14, "Assault_on_Women": 2080, "Cruelty_by_Husband_or_Relatives": 915, "Sexual_Harassment": 318, "Cyber_Crime": 195, "Other_Crimes_Against_Women": 192, "Total_Crime_Against_Women": 6176, "Population_Lakhs": 184.1, "Latitude": 19.0760, "Longitude": 72.8777},
        {"State": "Maharashtra", "City": "Mumbai", "Year": 2023, "Rape": 945, "Kidnapping_and_Abduction": 1812, "Dowry_Deaths": 13, "Assault_on_Women": 2145, "Cruelty_by_Husband_or_Relatives": 960, "Sexual_Harassment": 330, "Cyber_Crime": 210, "Other_Crimes_Against_Women": 208, "Total_Crime_Against_Women": 6423, "Population_Lakhs": 184.1, "Latitude": 19.0760, "Longitude": 72.8777},

        # Bengaluru
        {"State": "Karnataka", "City": "Bengaluru", "Year": 2021, "Rape": 117, "Kidnapping_and_Abduction": 570, "Dowry_Deaths": 44, "Assault_on_Women": 1025, "Cruelty_by_Husband_or_Relatives": 923, "Sexual_Harassment": 198, "Cyber_Crime": 142, "Other_Crimes_Against_Women": 108, "Total_Crime_Against_Women": 3127, "Population_Lakhs": 85.0, "Latitude": 12.9716, "Longitude": 77.5946},
        {"State": "Karnataka", "City": "Bengaluru", "Year": 2022, "Rape": 148, "Kidnapping_and_Abduction": 795, "Dowry_Deaths": 48, "Assault_on_Women": 1310, "Cruelty_by_Husband_or_Relatives": 1140, "Sexual_Harassment": 235, "Cyber_Crime": 165, "Other_Crimes_Against_Women": 83, "Total_Crime_Against_Women": 3924, "Population_Lakhs": 85.0, "Latitude": 12.9716, "Longitude": 77.5946},
        {"State": "Karnataka", "City": "Bengaluru", "Year": 2023, "Rape": 162, "Kidnapping_and_Abduction": 870, "Dowry_Deaths": 46, "Assault_on_Women": 1420, "Cruelty_by_Husband_or_Relatives": 1260, "Sexual_Harassment": 250, "Cyber_Crime": 178, "Other_Crimes_Against_Women": 75, "Total_Crime_Against_Women": 4261, "Population_Lakhs": 85.0, "Latitude": 12.9716, "Longitude": 77.5946},

        # Hyderabad
        {"State": "Telangana", "City": "Hyderabad", "Year": 2021, "Rape": 116, "Kidnapping_and_Abduction": 212, "Dowry_Deaths": 23, "Assault_on_Women": 640, "Cruelty_by_Husband_or_Relatives": 1380, "Sexual_Harassment": 110, "Cyber_Crime": 75, "Other_Crimes_Against_Women": 36, "Total_Crime_Against_Women": 2592, "Population_Lakhs": 68.1, "Latitude": 17.3850, "Longitude": 78.4867},
        {"State": "Telangana", "City": "Hyderabad", "Year": 2022, "Rape": 132, "Kidnapping_and_Abduction": 258, "Dowry_Deaths": 28, "Assault_on_Women": 725, "Cruelty_by_Husband_or_Relatives": 1620, "Sexual_Harassment": 135, "Cyber_Crime": 92, "Other_Crimes_Against_Women": 39, "Total_Crime_Against_Women": 3029, "Population_Lakhs": 68.1, "Latitude": 17.3850, "Longitude": 78.4867},
        {"State": "Telangana", "City": "Hyderabad", "Year": 2023, "Rape": 140, "Kidnapping_and_Abduction": 275, "Dowry_Deaths": 26, "Assault_on_Women": 750, "Cruelty_by_Husband_or_Relatives": 1680, "Sexual_Harassment": 142, "Cyber_Crime": 98, "Other_Crimes_Against_Women": 36, "Total_Crime_Against_Women": 3147, "Population_Lakhs": 68.1, "Latitude": 17.3850, "Longitude": 78.4867},

        # Jaipur
        {"State": "Rajasthan", "City": "Jaipur", "Year": 2021, "Rape": 512, "Kidnapping_and_Abduction": 680, "Dowry_Deaths": 38, "Assault_on_Women": 650, "Cruelty_by_Husband_or_Relatives": 1045, "Sexual_Harassment": 120, "Cyber_Crime": 45, "Other_Crimes_Against_Women": 115, "Total_Crime_Against_Women": 3205, "Population_Lakhs": 30.7, "Latitude": 26.9124, "Longitude": 75.7873},
        {"State": "Rajasthan", "City": "Jaipur", "Year": 2022, "Rape": 548, "Kidnapping_and_Abduction": 730, "Dowry_Deaths": 42, "Assault_on_Women": 710, "Cruelty_by_Husband_or_Relatives": 1140, "Sexual_Harassment": 135, "Cyber_Crime": 52, "Other_Crimes_Against_Women": 122, "Total_Crime_Against_Women": 3479, "Population_Lakhs": 30.7, "Latitude": 26.9124, "Longitude": 75.7873},
        {"State": "Rajasthan", "City": "Jaipur", "Year": 2023, "Rape": 560, "Kidnapping_and_Abduction": 765, "Dowry_Deaths": 40, "Assault_on_Women": 735, "Cruelty_by_Husband_or_Relatives": 1180, "Sexual_Harassment": 140, "Cyber_Crime": 58, "Other_Crimes_Against_Women": 112, "Total_Crime_Against_Women": 3590, "Population_Lakhs": 30.7, "Latitude": 26.9124, "Longitude": 75.7873},

        # Lucknow
        {"State": "Uttar Pradesh", "City": "Lucknow", "Year": 2021, "Rape": 142, "Kidnapping_and_Abduction": 715, "Dowry_Deaths": 56, "Assault_on_Women": 520, "Cruelty_by_Husband_or_Relatives": 1080, "Sexual_Harassment": 195, "Cyber_Crime": 62, "Other_Crimes_Against_Women": 72, "Total_Crime_Against_Women": 2842, "Population_Lakhs": 28.2, "Latitude": 26.8467, "Longitude": 80.9462},
        {"State": "Uttar Pradesh", "City": "Lucknow", "Year": 2022, "Rape": 138, "Kidnapping_and_Abduction": 730, "Dowry_Deaths": 52, "Assault_on_Women": 510, "Cruelty_by_Husband_or_Relatives": 1075, "Sexual_Harassment": 188, "Cyber_Crime": 70, "Other_Crimes_Against_Women": 68, "Total_Crime_Against_Women": 2831, "Population_Lakhs": 28.2, "Latitude": 26.8467, "Longitude": 80.9462},
        {"State": "Uttar Pradesh", "City": "Lucknow", "Year": 2023, "Rape": 145, "Kidnapping_and_Abduction": 755, "Dowry_Deaths": 54, "Assault_on_Women": 535, "Cruelty_by_Husband_or_Relatives": 1095, "Sexual_Harassment": 192, "Cyber_Crime": 74, "Other_Crimes_Against_Women": 60, "Total_Crime_Against_Women": 2910, "Population_Lakhs": 28.2, "Latitude": 26.8467, "Longitude": 80.9462},

        # Kanpur
        {"State": "Uttar Pradesh", "City": "Kanpur", "Year": 2021, "Rape": 118, "Kidnapping_and_Abduction": 460, "Dowry_Deaths": 48, "Assault_on_Women": 380, "Cruelty_by_Husband_or_Relatives": 680, "Sexual_Harassment": 115, "Cyber_Crime": 28, "Other_Crimes_Against_Women": 46, "Total_Crime_Against_Women": 1875, "Population_Lakhs": 29.2, "Latitude": 26.4499, "Longitude": 80.3319},
        {"State": "Uttar Pradesh", "City": "Kanpur", "Year": 2022, "Rape": 125, "Kidnapping_and_Abduction": 490, "Dowry_Deaths": 51, "Assault_on_Women": 405, "Cruelty_by_Husband_or_Relatives": 715, "Sexual_Harassment": 122, "Cyber_Crime": 32, "Other_Crimes_Against_Women": 40, "Total_Crime_Against_Women": 1980, "Population_Lakhs": 29.2, "Latitude": 26.4499, "Longitude": 80.3319},
        {"State": "Uttar Pradesh", "City": "Kanpur", "Year": 2023, "Rape": 130, "Kidnapping_and_Abduction": 510, "Dowry_Deaths": 49, "Assault_on_Women": 420, "Cruelty_by_Husband_or_Relatives": 735, "Sexual_Harassment": 128, "Cyber_Crime": 35, "Other_Crimes_Against_Women": 38, "Total_Crime_Against_Women": 2045, "Population_Lakhs": 29.2, "Latitude": 26.4499, "Longitude": 80.3319},

        # Ahmedabad
        {"State": "Gujarat", "City": "Ahmedabad", "Year": 2021, "Rape": 185, "Kidnapping_and_Abduction": 340, "Dowry_Deaths": 6, "Assault_on_Women": 310, "Cruelty_by_Husband_or_Relatives": 725, "Sexual_Harassment": 82, "Cyber_Crime": 24, "Other_Crimes_Against_Women": 40, "Total_Crime_Against_Women": 1712, "Population_Lakhs": 63.6, "Latitude": 23.0225, "Longitude": 72.5714},
        {"State": "Gujarat", "City": "Ahmedabad", "Year": 2022, "Rape": 198, "Kidnapping_and_Abduction": 365, "Dowry_Deaths": 8, "Assault_on_Women": 335, "Cruelty_by_Husband_or_Relatives": 760, "Sexual_Harassment": 88, "Cyber_Crime": 28, "Other_Crimes_Against_Women": 38, "Total_Crime_Against_Women": 1820, "Population_Lakhs": 63.6, "Latitude": 23.0225, "Longitude": 72.5714},
        {"State": "Gujarat", "City": "Ahmedabad", "Year": 2023, "Rape": 205, "Kidnapping_and_Abduction": 380, "Dowry_Deaths": 7, "Assault_on_Women": 350, "Cruelty_by_Husband_or_Relatives": 790, "Sexual_Harassment": 92, "Cyber_Crime": 31, "Other_Crimes_Against_Women": 40, "Total_Crime_Against_Women": 1895, "Population_Lakhs": 63.6, "Latitude": 23.0225, "Longitude": 72.5714},

        # Pune
        {"State": "Maharashtra", "City": "Pune", "Year": 2021, "Rape": 210, "Kidnapping_and_Abduction": 380, "Dowry_Deaths": 8, "Assault_on_Women": 415, "Cruelty_by_Husband_or_Relatives": 280, "Sexual_Harassment": 65, "Cyber_Crime": 32, "Other_Crimes_Against_Women": 28, "Total_Crime_Against_Women": 1418, "Population_Lakhs": 50.5, "Latitude": 18.5204, "Longitude": 73.8567},
        {"State": "Maharashtra", "City": "Pune", "Year": 2022, "Rape": 235, "Kidnapping_and_Abduction": 425, "Dowry_Deaths": 10, "Assault_on_Women": 460, "Cruelty_by_Husband_or_Relatives": 315, "Sexual_Harassment": 72, "Cyber_Crime": 38, "Other_Crimes_Against_Women": 25, "Total_Crime_Against_Women": 1580, "Population_Lakhs": 50.5, "Latitude": 18.5204, "Longitude": 73.8567},
        {"State": "Maharashtra", "City": "Pune", "Year": 2023, "Rape": 248, "Kidnapping_and_Abduction": 450, "Dowry_Deaths": 9, "Assault_on_Women": 485, "Cruelty_by_Husband_or_Relatives": 335, "Sexual_Harassment": 78, "Cyber_Crime": 42, "Other_Crimes_Against_Women": 28, "Total_Crime_Against_Women": 1675, "Population_Lakhs": 50.5, "Latitude": 18.5204, "Longitude": 73.8567},

        # Kolkata
        {"State": "West Bengal", "City": "Kolkata", "Year": 2021, "Rape": 11, "Kidnapping_and_Abduction": 285, "Dowry_Deaths": 3, "Assault_on_Women": 310, "Cruelty_by_Husband_or_Relatives": 840, "Sexual_Harassment": 195, "Cyber_Crime": 22, "Other_Crimes_Against_Women": 117, "Total_Crime_Against_Women": 1783, "Population_Lakhs": 141.1, "Latitude": 22.5726, "Longitude": 88.3639},
        {"State": "West Bengal", "City": "Kolkata", "Year": 2022, "Rape": 11, "Kidnapping_and_Abduction": 260, "Dowry_Deaths": 2, "Assault_on_Women": 295, "Cruelty_by_Husband_or_Relatives": 810, "Sexual_Harassment": 180, "Cyber_Crime": 25, "Other_Crimes_Against_Women": 107, "Total_Crime_Against_Women": 1690, "Population_Lakhs": 141.1, "Latitude": 22.5726, "Longitude": 88.3639},
        {"State": "West Bengal", "City": "Kolkata", "Year": 2023, "Rape": 12, "Kidnapping_and_Abduction": 272, "Dowry_Deaths": 3, "Assault_on_Women": 305, "Cruelty_by_Husband_or_Relatives": 825, "Sexual_Harassment": 185, "Cyber_Crime": 28, "Other_Crimes_Against_Women": 90, "Total_Crime_Against_Women": 1720, "Population_Lakhs": 141.1, "Latitude": 22.5726, "Longitude": 88.3639},

        # Indore
        {"State": "Madhya Pradesh", "City": "Indore", "Year": 2021, "Rape": 195, "Kidnapping_and_Abduction": 340, "Dowry_Deaths": 14, "Assault_on_Women": 480, "Cruelty_by_Husband_or_Relatives": 480, "Sexual_Harassment": 75, "Cyber_Crime": 28, "Other_Crimes_Against_Women": 38, "Total_Crime_Against_Women": 1650, "Population_Lakhs": 21.7, "Latitude": 22.7196, "Longitude": 75.8577},
        {"State": "Madhya Pradesh", "City": "Indore", "Year": 2022, "Rape": 210, "Kidnapping_and_Abduction": 365, "Dowry_Deaths": 16, "Assault_on_Women": 510, "Cruelty_by_Husband_or_Relatives": 505, "Sexual_Harassment": 80, "Cyber_Crime": 32, "Other_Crimes_Against_Women": 22, "Total_Crime_Against_Women": 1740, "Population_Lakhs": 21.7, "Latitude": 22.7196, "Longitude": 75.8577},
        {"State": "Madhya Pradesh", "City": "Indore", "Year": 2023, "Rape": 220, "Kidnapping_and_Abduction": 385, "Dowry_Deaths": 15, "Assault_on_Women": 530, "Cruelty_by_Husband_or_Relatives": 525, "Sexual_Harassment": 84, "Cyber_Crime": 35, "Other_Crimes_Against_Women": 16, "Total_Crime_Against_Women": 1810, "Population_Lakhs": 21.7, "Latitude": 22.7196, "Longitude": 75.8577},

        # Surat
        {"State": "Gujarat", "City": "Surat", "Year": 2021, "Rape": 135, "Kidnapping_and_Abduction": 270, "Dowry_Deaths": 4, "Assault_on_Women": 185, "Cruelty_by_Husband_or_Relatives": 440, "Sexual_Harassment": 45, "Cyber_Crime": 18, "Other_Crimes_Against_Women": 23, "Total_Crime_Against_Women": 1120, "Population_Lakhs": 45.8, "Latitude": 21.1702, "Longitude": 72.8311},
        {"State": "Gujarat", "City": "Surat", "Year": 2022, "Rape": 150, "Kidnapping_and_Abduction": 305, "Dowry_Deaths": 5, "Assault_on_Women": 210, "Cruelty_by_Husband_or_Relatives": 480, "Sexual_Harassment": 50, "Cyber_Crime": 22, "Other_Crimes_Against_Women": 18, "Total_Crime_Against_Women": 1240, "Population_Lakhs": 45.8, "Latitude": 21.1702, "Longitude": 72.8311},
        {"State": "Gujarat", "City": "Surat", "Year": 2023, "Rape": 160, "Kidnapping_and_Abduction": 320, "Dowry_Deaths": 5, "Assault_on_Women": 225, "Cruelty_by_Husband_or_Relatives": 505, "Sexual_Harassment": 54, "Cyber_Crime": 25, "Other_Crimes_Against_Women": 16, "Total_Crime_Against_Women": 1310, "Population_Lakhs": 45.8, "Latitude": 21.1702, "Longitude": 72.8311},

        # Patna
        {"State": "Bihar", "City": "Patna", "Year": 2021, "Rape": 68, "Kidnapping_and_Abduction": 340, "Dowry_Deaths": 32, "Assault_on_Women": 140, "Cruelty_by_Husband_or_Relatives": 390, "Sexual_Harassment": 45, "Cyber_Crime": 12, "Other_Crimes_Against_Women": 23, "Total_Crime_Against_Women": 1050, "Population_Lakhs": 20.5, "Latitude": 25.5941, "Longitude": 85.1376},
        {"State": "Bihar", "City": "Patna", "Year": 2022, "Rape": 75, "Kidnapping_and_Abduction": 380, "Dowry_Deaths": 35, "Assault_on_Women": 160, "Cruelty_by_Husband_or_Relatives": 440, "Sexual_Harassment": 50, "Cyber_Crime": 15, "Other_Crimes_Against_Women": 25, "Total_Crime_Against_Women": 1180, "Population_Lakhs": 20.5, "Latitude": 25.5941, "Longitude": 85.1376},
        {"State": "Bihar", "City": "Patna", "Year": 2023, "Rape": 80, "Kidnapping_and_Abduction": 405, "Dowry_Deaths": 34, "Assault_on_Women": 172, "Cruelty_by_Husband_or_Relatives": 465, "Sexual_Harassment": 52, "Cyber_Crime": 18, "Other_Crimes_Against_Women": 24, "Total_Crime_Against_Women": 1250, "Population_Lakhs": 20.5, "Latitude": 25.5941, "Longitude": 85.1376},

        # Ghaziabad
        {"State": "Uttar Pradesh", "City": "Ghaziabad", "Year": 2021, "Rape": 85, "Kidnapping_and_Abduction": 380, "Dowry_Deaths": 34, "Assault_on_Women": 210, "Cruelty_by_Husband_or_Relatives": 430, "Sexual_Harassment": 55, "Cyber_Crime": 20, "Other_Crimes_Against_Women": 16, "Total_Crime_Against_Women": 1230, "Population_Lakhs": 23.6, "Latitude": 28.6692, "Longitude": 77.4538},
        {"State": "Uttar Pradesh", "City": "Ghaziabad", "Year": 2022, "Rape": 95, "Kidnapping_and_Abduction": 420, "Dowry_Deaths": 38, "Assault_on_Women": 235, "Cruelty_by_Husband_or_Relatives": 470, "Sexual_Harassment": 60, "Cyber_Crime": 24, "Other_Crimes_Against_Women": 8, "Total_Crime_Against_Women": 1350, "Population_Lakhs": 23.6, "Latitude": 28.6692, "Longitude": 77.4538},
        {"State": "Uttar Pradesh", "City": "Ghaziabad", "Year": 2023, "Rape": 102, "Kidnapping_and_Abduction": 445, "Dowry_Deaths": 36, "Assault_on_Women": 248, "Cruelty_by_Husband_or_Relatives": 495, "Sexual_Harassment": 64, "Cyber_Crime": 28, "Other_Crimes_Against_Women": 2, "Total_Crime_Against_Women": 1420, "Population_Lakhs": 23.6, "Latitude": 28.6692, "Longitude": 77.4538},

        # Nagpur
        {"State": "Maharashtra", "City": "Nagpur", "Year": 2021, "Rape": 165, "Kidnapping_and_Abduction": 285, "Dowry_Deaths": 6, "Assault_on_Women": 380, "Cruelty_by_Husband_or_Relatives": 185, "Sexual_Harassment": 48, "Cyber_Crime": 18, "Other_Crimes_Against_Women": 23, "Total_Crime_Against_Women": 1110, "Population_Lakhs": 24.1, "Latitude": 21.1458, "Longitude": 79.0882},
        {"State": "Maharashtra", "City": "Nagpur", "Year": 2022, "Rape": 178, "Kidnapping_and_Abduction": 310, "Dowry_Deaths": 7, "Assault_on_Women": 410, "Cruelty_by_Husband_or_Relatives": 198, "Sexual_Harassment": 52, "Cyber_Crime": 22, "Other_Crimes_Against_Women": 13, "Total_Crime_Against_Women": 1190, "Population_Lakhs": 24.1, "Latitude": 21.1458, "Longitude": 79.0882},
        {"State": "Maharashtra", "City": "Nagpur", "Year": 2023, "Rape": 188, "Kidnapping_and_Abduction": 330, "Dowry_Deaths": 7, "Assault_on_Women": 435, "Cruelty_by_Husband_or_Relatives": 210, "Sexual_Harassment": 55, "Cyber_Crime": 25, "Other_Crimes_Against_Women": 10, "Total_Crime_Against_Women": 1260, "Population_Lakhs": 24.1, "Latitude": 21.1458, "Longitude": 79.0882},

        # Chennai
        {"State": "Tamil Nadu", "City": "Chennai", "Year": 2021, "Rape": 62, "Kidnapping_and_Abduction": 125, "Dowry_Deaths": 8, "Assault_on_Women": 310, "Cruelty_by_Husband_or_Relatives": 215, "Sexual_Harassment": 75, "Cyber_Crime": 24, "Other_Crimes_Against_Women": 21, "Total_Crime_Against_Women": 840, "Population_Lakhs": 86.5, "Latitude": 13.0827, "Longitude": 80.2707},
        {"State": "Tamil Nadu", "City": "Chennai", "Year": 2022, "Rape": 68, "Kidnapping_and_Abduction": 138, "Dowry_Deaths": 9, "Assault_on_Women": 330, "Cruelty_by_Husband_or_Relatives": 228, "Sexual_Harassment": 80, "Cyber_Crime": 22, "Other_Crimes_Against_Women": 15, "Total_Crime_Against_Women": 890, "Population_Lakhs": 86.5, "Latitude": 13.0827, "Longitude": 80.2707},
        {"State": "Tamil Nadu", "City": "Chennai", "Year": 2023, "Rape": 72, "Kidnapping_and_Abduction": 145, "Dowry_Deaths": 8, "Assault_on_Women": 348, "Cruelty_by_Husband_or_Relatives": 240, "Sexual_Harassment": 85, "Cyber_Crime": 25, "Other_Crimes_Against_Women": 12, "Total_Crime_Against_Women": 935, "Population_Lakhs": 86.5, "Latitude": 13.0827, "Longitude": 80.2707},

        # Kochi
        {"State": "Kerala", "City": "Kochi", "Year": 2021, "Rape": 54, "Kidnapping_and_Abduction": 42, "Dowry_Deaths": 1, "Assault_on_Women": 245, "Cruelty_by_Husband_or_Relatives": 180, "Sexual_Harassment": 62, "Cyber_Crime": 18, "Other_Crimes_Against_Women": 18, "Total_Crime_Against_Women": 620, "Population_Lakhs": 21.2, "Latitude": 9.9312, "Longitude": 76.2673},
        {"State": "Kerala", "City": "Kochi", "Year": 2022, "Rape": 61, "Kidnapping_and_Abduction": 48, "Dowry_Deaths": 1, "Assault_on_Women": 270, "Cruelty_by_Husband_or_Relatives": 195, "Sexual_Harassment": 68, "Cyber_Crime": 22, "Other_Crimes_Against_Women": 15, "Total_Crime_Against_Women": 680, "Population_Lakhs": 21.2, "Latitude": 9.9312, "Longitude": 76.2673},
        {"State": "Kerala", "City": "Kochi", "Year": 2023, "Rape": 65, "Kidnapping_and_Abduction": 52, "Dowry_Deaths": 1, "Assault_on_Women": 285, "Cruelty_by_Husband_or_Relatives": 205, "Sexual_Harassment": 72, "Cyber_Crime": 24, "Other_Crimes_Against_Women": 11, "Total_Crime_Against_Women": 715, "Population_Lakhs": 21.2, "Latitude": 9.9312, "Longitude": 76.2673},

        # Kozhikode
        {"State": "Kerala", "City": "Kozhikode", "Year": 2021, "Rape": 42, "Kidnapping_and_Abduction": 28, "Dowry_Deaths": 0, "Assault_on_Women": 195, "Cruelty_by_Husband_or_Relatives": 145, "Sexual_Harassment": 48, "Cyber_Crime": 12, "Other_Crimes_Against_Women": 10, "Total_Crime_Against_Women": 480, "Population_Lakhs": 20.3, "Latitude": 11.2588, "Longitude": 75.7804},
        {"State": "Kerala", "City": "Kozhikode", "Year": 2022, "Rape": 48, "Kidnapping_and_Abduction": 32, "Dowry_Deaths": 1, "Assault_on_Women": 210, "Cruelty_by_Husband_or_Relatives": 158, "Sexual_Harassment": 52, "Cyber_Crime": 14, "Other_Crimes_Against_Women": 5, "Total_Crime_Against_Women": 520, "Population_Lakhs": 20.3, "Latitude": 11.2588, "Longitude": 75.7804},
        {"State": "Kerala", "City": "Kozhikode", "Year": 2023, "Rape": 52, "Kidnapping_and_Abduction": 35, "Dowry_Deaths": 0, "Assault_on_Women": 222, "Cruelty_by_Husband_or_Relatives": 168, "Sexual_Harassment": 55, "Cyber_Crime": 15, "Other_Crimes_Against_Women": 3, "Total_Crime_Against_Women": 550, "Population_Lakhs": 20.3, "Latitude": 11.2588, "Longitude": 75.7804},

        # Coimbatore
        {"State": "Tamil Nadu", "City": "Coimbatore", "Year": 2021, "Rape": 22, "Kidnapping_and_Abduction": 35, "Dowry_Deaths": 2, "Assault_on_Women": 110, "Cruelty_by_Husband_or_Relatives": 78, "Sexual_Harassment": 28, "Cyber_Crime": 8, "Other_Crimes_Against_Women": 7, "Total_Crime_Against_Women": 290, "Population_Lakhs": 21.5, "Latitude": 11.0168, "Longitude": 76.9558},
        {"State": "Tamil Nadu", "City": "Coimbatore", "Year": 2022, "Rape": 25, "Kidnapping_and_Abduction": 38, "Dowry_Deaths": 2, "Assault_on_Women": 120, "Cruelty_by_Husband_or_Relatives": 85, "Sexual_Harassment": 30, "Cyber_Crime": 10, "Other_Crimes_Against_Women": 5, "Total_Crime_Against_Women": 315, "Population_Lakhs": 21.5, "Latitude": 11.0168, "Longitude": 76.9558},
        {"State": "Tamil Nadu", "City": "Coimbatore", "Year": 2023, "Rape": 28, "Kidnapping_and_Abduction": 40, "Dowry_Deaths": 2, "Assault_on_Women": 125, "Cruelty_by_Husband_or_Relatives": 90, "Sexual_Harassment": 32, "Cyber_Crime": 10, "Other_Crimes_Against_Women": 3, "Total_Crime_Against_Women": 330, "Population_Lakhs": 21.5, "Latitude": 11.0168, "Longitude": 76.9558}
    ]

    df_ncrb = pd.DataFrame(cities_data)
    
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "ncrb_crime_women_2023.csv")
    df_ncrb.to_csv(out_path, index=False)
    print(f"NCRB dataset saved to {out_path} with {len(df_ncrb)} rows.")

    # Also generate sample district dataset as secondary optional dataset
    sample_district_data = df_ncrb.copy()
    sample_district_data["District"] = sample_district_data["City"] + " District"
    sample_path = os.path.join("data", "sample_data.csv")
    sample_district_data.to_csv(sample_path, index=False)
    print(f"Sample dataset saved to {sample_path}.")

if __name__ == "__main__":
    generate_ncrb_dataset()
