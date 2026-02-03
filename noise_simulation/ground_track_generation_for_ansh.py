import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


class groundtrack_interpolation:
    def interpolation(self, df, distance):
        # Remove duplicate points based on easting and northing
        df_unique = df.drop_duplicates(subset=['easting', 'northing'])

        
        # Calculate the differences between consecutive points
        diffs = np.sqrt(np.diff(df_unique['easting'])**2 + np.diff(df_unique['northing'])**2)

        # Calculate the average distance
        avg_distance = np.mean(diffs) if len(diffs) > 0 else 0

        # Cumulative distance along the path
        distances = np.insert(np.cumsum(diffs), 0, 0)

        # Create lists to hold the new easting and northing values
        new_easting = []
        new_northing = []

        # Iterate through the points to check for large gaps
        for i in range(len(diffs)):
            start_point = df_unique.iloc[i]
            new_easting.append(start_point['easting'])
            new_northing.append(start_point['northing'])

            if i < len(diffs) - 1:
                end_point = df_unique.iloc[i + 1]
                gap_distance = diffs[i]

                # Check if the gap is more than 5 times the average distance
                if gap_distance > 5 * avg_distance:
                    # Calculate the number of new points to insert
                    num_new_points = int(np.ceil(gap_distance / avg_distance)) - 1
                    
                    # Generate new points in a straight line
                    for j in range(1, num_new_points + 1):
                        ratio = j / (num_new_points + 1)
                        new_easting.append(start_point['easting'] + ratio * (end_point['easting'] - start_point['easting']))
                        new_northing.append(start_point['northing'] + ratio * (end_point['northing'] - start_point['northing']))

        # Convert the new points to a DataFrame
        df_new_points = pd.DataFrame({
            'easting': new_easting,
            'northing': new_northing
        })

        # Cumulative distance along the new path
        new_diffs = np.sqrt(np.diff(df_new_points['easting'])**2 + np.diff(df_new_points['northing'])**2)
        new_distances = np.insert(np.cumsum(new_diffs), 0, 0)

        # New distance values for interpolation
        interval = distance
        new_distances_interp = np.arange(0, new_distances[-1], interval)

        
        
        # Cubic spline interpolation
        spline_easting = CubicSpline(new_distances, df_new_points['easting'])
        spline_northing = CubicSpline(new_distances, df_new_points['northing'])

        # Interpolated values
        interp_easting = spline_easting(new_distances_interp)
        interp_northing = spline_northing(new_distances_interp)

        # Create a new dataframe with interpolated values
        df_interp = pd.DataFrame({
            'easting': interp_easting,
            'northing': interp_northing
        })

        #Plot the results
        # plt.scatter(df_unique['easting'], df_unique['northing'], color='blue', label='Original Points')
        # plt.plot(df_interp['easting'], df_interp['northing'], color='red', label='Interpolated Path')
        # plt.scatter(df_interp['easting'], df_interp['northing'], color='red')
        # plt.legend()
        # plt.show()

        return df_interp

    def calculate_distance(self, p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def calculate_area(self, p1, p2, p3):
        # Using the shoelace formula for the area of a triangle
        return abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0

    def calculate_turn_radius(self, df):
        # Initialize columns for turn radius and direction
        df['turn_radius'] = 0.0
        df['turn_direction'] = "'s'"

        for i in range(1, len(df) - 1):
            p1 = df.iloc[i - 1][['easting', 'northing']].values
            p2 = df.iloc[i][['easting', 'northing']].values
            p3 = df.iloc[i + 1][['easting', 'northing']].values


            # Calculate distances between points
            a = self.calculate_distance(p1, p2)
            b = self.calculate_distance(p2, p3)
            c = self.calculate_distance(p1, p3)

            
            # Calculate area K of the triangle formed by points p1, p2, and p3
            K = self.calculate_area(p1, p2, p3)


            # Avoid division by zero
            if K == 0:
                r = 0  # or some other handling for undefined radius
                direction = "'s'"  # Set to straight if collinear
            else:
                r = (a * b * c) / (4 * K)
                
                # Compute the vectors
                vec1 = p2 - p1
                vec2 = p3 - p2
                
                # Calculate the cross product
                cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
                
                # Determine if the curve is to the left or right
                if cross_product > 0:
                    direction = "'l'"
                elif cross_product < 0:
                    direction = "'r'"
                else:
                    direction = "'s'"  # Additional handling for collinear points

            # Assign the radius and direction to the DataFrame
            if direction == "'l'" or direction =="'r'":
                df.loc[i, 'turn_radius'] = r
                df.loc[i, 'turn_direction'] = direction
            else:
                df.loc[i, 'turn_radius'] = 0
                df.loc[i, 'turn_direction'] = "'s'"

        return df

if __name__ == "__main__":
    # the result of this function is a gorund track that can be used for Doc. 29 modeling


    # can be changed to modify the interpolation length
    interpolation_length = 200  # meters
    # here you insert the name of your groundtrack file, so the result of your clustering
    raw_groundtrack = 'groundtrack_example.csv'
    # specify runway and mode to get the correct starting point
    runway_key = '27L'
    mode = 'landing'  # 'landing' or 'take-off'
    # specify name for output file
    name = "cluster_1"

    # this function selects the correct starting point based on runway and mode (to make sure the groundtrack starts at the runway threshold)
    if '27L' in runway_key or '09R' in runway_key:
        if mode == 'landing' and '27L' in runway_key or mode == 'take-off' and '09R' in runway_key:
            starting_place_x = 546026.9
            starting_place_y = 5811859.4
        elif mode == 'landing' and '09R' in runway_key or mode == 'take-off' and '27L' in runway_key:
            starting_place_x = 548229.4
            starting_place_y = 5811780.4
                            
    elif '09L' in runway_key or '27R' in runway_key:
        if mode == 'landing' and '27R' in runway_key or mode == 'take-off' and '09L' in runway_key:
            starting_place_x = 544424.6
            starting_place_y = 5813317.4
        elif mode == 'landing' and '09L' in runway_key or mode == 'take-off' and '27R' in runway_key:
            starting_place_x = 547474.3
            starting_place_y = 5813208.8


    groundtrack_df = pd.read_csv(raw_groundtrack, sep=';')
    # New row to be added
    groundtrack_df.drop(index=0, inplace=True)
    new_row = pd.DataFrame({'Unnamed: 0': [0], 'easting': [starting_place_x], 'northing': [starting_place_y]})
        
    # Concatenate the new row before the existing DataFrame
    groundtrack_df = pd.concat([new_row, groundtrack_df], ignore_index=True)
    groundtrack = groundtrack_interpolation()
    df_interpolated = groundtrack.interpolation(groundtrack_df, interpolation_length)
            
    # Calculate turn radius based on the interpolated data
    df_interpolated = groundtrack.calculate_turn_radius(df_interpolated)
    #print(df_interpolated)
    # Step 1: Rename the 'turn_radius' column to 'radius'
    df_interpolated.rename(columns={'turn_radius': 'radius'}, inplace=True)
    # Step 2: Calculate the distance s
    # Calculate the distance between consecutive points
    distances = np.sqrt((df_interpolated['easting'].diff()**2) + (df_interpolated['northing'].diff()**2))
    # Fill the first distance (for row 0) with 0 and then calculate the cumulative sum
    df_interpolated['s'] = distances.cumsum().fillna(0)
    #print(df_interpolated)
    # cut the ground path to examine influence of its length
    #print(df_interpolated)
    groundtrack_file = f'groundtrack_discr_{name}.csv'
    df_interpolated.to_csv(groundtrack_file, sep=';')
