import csv
import pandas as pd
import numpy as np

path = '/Users/parikshitshah/Dropbox/'
filename = "venkat_pari_data_dump.csv"


def load_data(path, filename):
	filename = path + filename
	print(filename)
	df = getstuff(filename)
	df["system_running_status"] = df["bop_plc_system_running"].astype(str).str.strip().map(lambda x: 1 if x.lower() == "t" else 0)
	df = df[::100]
	# Compute the windowed derivatives
	windowed_derivatives_df = windowed_derivative(df, columns_to_compute, window_size=100)


	for column in columns_to_compute:
    		mean = windowed_derivatives_df[column].mean()
    		std =  windowed_derivatives_df[column].std()
    		colname = column + "_norm"
    		windowed_derivatives_df[colname] = (windowed_derivatives_df[column] - mean) / std

    # Add the timestamp column to the result
	windowed_derivatives_df['timestamp'] = df['timestamp']
	# 'system_running_status'
	windowed_derivatives_df['system_running_status'] = df['system_running_status']
	return windowed_derivatives_df

def getstuff(filename):
    df =pd.read_csv(filename, skiprows = range(1,1252000,2) )
    return df

# compute windowed derivatives

def windowed_derivative(df, columns, window_size=100):
    # Create an empty dataframe to store the windowed derivatives
    windowed_derivatives = pd.DataFrame(index=df.index, columns=columns)

    for column in columns:
        # Create an empty list to store the windowed derivatives for each row
        windowed_derivative_column = []

        for i in range(len(df)):
            # Calculate the indices for the previous and next windows
            prev_window = df[column].iloc[max(0, i - window_size):i]  # previous 100 rows
            next_window = df[column].iloc[i + 1:i + 1 + window_size]  # next 100 rows

            # Compute the mean of the previous and next windows
            prev_mean = prev_window.mean() if len(prev_window) > 0 else np.nan
            next_mean = next_window.mean() if len(next_window) > 0 else np.nan

            # Compute the difference between the averages (windowed derivative)
            windowed_derivative_column.append(prev_mean - next_mean)

        # Store the result for the current column
        windowed_derivatives[column] = windowed_derivative_column

    return windowed_derivatives

columns_to_compute = [
    'bop_plc_inlet_acc_volume_mcf', 'bop_plc_inlet_ch4', 'bop_plc_inlet_co2',
    'bop_plc_inlet_flow', 'bop_plc_inlet_h2s', 'bop_plc_inlet_n2', 'bop_plc_inlet_o2',
    'bop_plc_inlet_pressure', 'bop_plc_inlet_temp', 'bop_plc_inlet_today_energy_mbtu',
    'bop_plc_inlet_today_volume_mcf', 'bop_plc_inlet_yest_energy_mbtu',
    'bop_plc_inlet_yest_volume_mcf', 'bop_plc_membrane_outlet_acc_volume_mcf',
    'bop_plc_membrane_outlet_ch4', 'bop_plc_membrane_outlet_co2', 'bop_plc_membrane_outlet_flow',
    'bop_plc_membrane_outlet_h2s', 'bop_plc_membrane_outlet_o2', 'bop_plc_membrane_outlet_pressure',
    'bop_plc_membrane_outlet_temp', 'bop_plc_membrane_outlet_today_energy_mbtu',
    'bop_plc_membrane_outlet_today_volume_mcf_real', 'bop_plc_membrane_outlet_yest_energy_mbtu',
    'bop_plc_membrane_outlet_yest_volume_mcf', 'bop_plc_plant_outlet_ch4', 'bop_plc_plant_outlet_co2',
    'bop_plc_plant_outlet_energy_btu', 'bop_plc_plant_outlet_flow', 'bop_plc_plant_outlet_h2o',
    'bop_plc_plant_outlet_h2s', 'bop_plc_plant_outlet_n2', 'bop_plc_plant_outlet_o2',
    'bop_plc_plant_outlet_pressure', 'bop_plc_plant_outlet_temp', 'bop_plc_metering_skid_delta_ch4',
    'bop_plc_metering_skid_delta_co2', 'bop_plc_metering_skid_delta_n2', 'bop_plc_metering_skid_delta_h2s',
    'bop_plc_metering_skid_delta_o2', 'bop_plc_metering_skid_delta_h2o', 'bop_plc_metering_skid_delta_energy_btu',
    'bop_plc_metering_skid_delta_flow', 'bop_plc_metering_skid_delta_pressure', 'bop_plc_metering_skid_delta_temp',
    'bems_plc_waste', 'bop_plc_membrane_outlet_n2', 'bop_plc_metering_skid_current_mmbtu',
    'bop_plc_plant_outlet_current_mmbtu', 'bop_plc_aptim_flare_today_flow_scfm', 'bop_plc_fe2109_today_flow_scfm',
    'bop_plc_fe2403_fit388_flow_scfm', 'bop_plc_fe2501_today_flow_scfm', 'bop_plc_fe2403_today_flow_scfm',
    'bop_plc_fit388_today_flow_scfm', 'bop_plc_methane_recovery_guild', 'bop_plc_methane_recovery_membrane',
    'bop_plc_methane_recovery_plant', 'bop_plc_power_meter_sb1_kwh', 'bop_plc_power_meter_sb2_kwh',
    'bop_plc_power_meter_sb3_kwh', 'bop_plc_power_meter_sb4_kwh', 'bop_plc_tox_s1_ati1501b_methane',
    'bop_plc_tox_s2_ati1501a_methane', 'bop_plc_at2503_flare_inlet_methane', 'bop_plc_methane_flow_recovery_guild',
    'bop_plc_methane_flow_recovery_membrane', 'bop_plc_methane_flow_recovery_plant'
]

