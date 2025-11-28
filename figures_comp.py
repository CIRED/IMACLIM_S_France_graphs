import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================
# PARAMÈTRES À CONFIGURER PAR L'UTILISATEUR
# ============================================

# Répertoire de base contenant les fichiers de simulation
base_dir = Path(r"c:\Users\douda\Documents\code_figures\chocs_cumules\variations_constraint_mx\b_sensi_elast_xm")

# MODE D'ANALYSE : 'single_year' ou 'temporal'
analysis_mode = 'temporal'  # 'single_year' : analyse pour une année donnée
                               # 'temporal' : évolution temporelle (X devient l'année)

# COMPARAISON AMS vs AME (optionnel)
# Options : None, 'ams_ame_difference', 'ams_ame_percentage'
# None : affiche uniquement les données AMS (comportement par défaut)
# 'ams_ame_difference' : affiche la différence absolue (AMS - AME)
# 'ams_ame_percentage' : affiche la différence relative en % ((AMS - AME) / AME * 100)
compare_scenarios = 'ams_ame_percentage'  # Mettre None pour désactiver la comparaison

# Variables à extraire et afficher
x_variable = 'coeff_constraint'  # Variable en abscisse (ignoré si analysis_mode='temporal')
y_variable = 'Real GDP Paas'        # Variable en ordonnée

# Pour mode 'single_year' : Nom de la colonne temporelle dans les fichiers CSV
time_column = 'values_2030'  # Peut être 'values_2018', 'values_2030', etc.

# Pour mode 'temporal' : Liste des années à extraire
temporal_columns = ['values_2018', 'values_2030', 'values_2040', 'values_2050']  # Liste des colonnes temporelles

# OPÉRATION SUR LA VARIABLE Y (optionnel)
# Options : None, 'difference', 'ratio', 'percentage_change'
# 'difference' : y_var2 - y_var1 (ex: variable_2030 - variable_2018)
# 'ratio' : y_var2 / y_var1
# 'percentage_change' : (y_var2 - y_var1) / y_var1 * 100
y_operation = None # Mettre None pour pas d'opération
y_variable_2 = 'Nominal Trade Balance'  # Deuxième variable pour l'opération (si y_operation n'est pas None)

# Paramètres de simulation pour distinguer les courbes
color_parameter = 'VAR_sigma_X'   # Paramètre qui définira les couleurs
linestyle_parameter = 'VAR_sigma_M'   # Paramètre qui définira les types de lignes
marker_parameter = None  # Paramètre qui définira les types de marqueurs (None pour désactiver)

# Titre et labels du graphique (générés automatiquement si None)
plot_title = ' '
x_label = None
y_label = "Variation relative du PIB (%)"

# FORMAT DE L'AXE Y
# Options : 'absolute' ou 'percentage'
# 'absolute' : affiche les valeurs telles quelles
# 'percentage' : multiplie par 100 et ajoute le symbole %
y_axis_format = 'absolute'  # 'absolute' ou 'percentage'

# AFFICHAGE DES DONNÉES THREEME
# Si True, affiche la courbe ThreeME en noir (en mode temporal avec comparaison AMS vs AME)
show_threeme = False  # True pour afficher ThreeME, False sinon

# Données ThreeME (variation en % par rapport à AME pour compare_scenarios='ams_ame_percentage'
#                  ou valeurs absolues selon le contexte)
# Format : {année: valeur}
# Exemple pour Real GDP Lasp:
threeme_data = {
    '2030': 1.99,
    '2040': 1.3,
    '2050': 1.3
}
# SAUVEGARDE DES DONNÉES EN CSV
save_csv = True  # True pour sauvegarder les données en CSV, False sinon
csv_filename = None  # Nom du fichier CSV (auto-généré si None)

# ============================================
# FIN DES PARAMÈTRES UTILISATEUR
# ============================================

# Find all FullTemplate files (AMS and optionally AME)
fulltemplate_files_ams = list(base_dir.glob("**/FullTemplate_AMSrun3mixnote_*.csv"))
fulltemplate_files_ame = list(base_dir.glob("**/FullTemplate_AMErun3dgtnote_*.csv")) if compare_scenarios else []

print(f"\nTotal AMS files found: {len(fulltemplate_files_ams)}")
if compare_scenarios:
    print(f"Total AME files found: {len(fulltemplate_files_ame)}")

# Storage for data
data_ams = []
data_ame = []

# Determine which variables to extract based on mode
if analysis_mode == 'temporal':
    # For temporal analysis, we need y_variable for all time periods
    variables_to_extract = [y_variable, color_parameter, linestyle_parameter]
    if marker_parameter:
        variables_to_extract.append(marker_parameter)
    if y_operation and y_variable_2:
        variables_to_extract.append(y_variable_2)
    columns_to_extract = temporal_columns
else:
    # For single year analysis, extract x_variable, y_variable and parameters
    variables_to_extract = [x_variable, y_variable, color_parameter, linestyle_parameter]
    if marker_parameter:
        variables_to_extract.append(marker_parameter)
    if y_operation and y_variable_2:
        variables_to_extract.append(y_variable_2)
    columns_to_extract = [time_column]

# Function to read and extract data from files
def extract_data_from_files(file_list, scenario_type):
    """Extract data from FullTemplate files"""
    data = []
    for file_path in file_list:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, sep=';', decimal=',')
            
            # Debug: print available variables for first file
            if len(data) == 0:
                print(f"\nDEBUG - Variables disponibles dans {file_path.name}:")
                print(f"Total variables: {len(df['Variables'].unique())}")
                # Check if our variables exist
                for var_name in variables_to_extract:
                    if var_name and var_name in df['Variables'].values:
                        print(f"  ✓ {var_name}: FOUND")
                    elif var_name:
                        print(f"  ✗ {var_name}: NOT FOUND")
                print(f"\nColonnes disponibles: {list(df.columns)}")
                print(f"\nPremières variables:")
                print(df['Variables'].head(20).tolist())
            
            # Dictionary to store extracted values
            extracted_data = {'file': file_path.name, 'scenario_type': scenario_type}
            
            # Extract scenario name and simulation number from filename
            filename = file_path.name
            if 'FullTemplate_' in filename:
                scenario_part = filename.replace('FullTemplate_', '').replace('.csv', '')
                extracted_data['scenario'] = scenario_part  # e.g., "AMSrun3mix_1"
                
                # Extract simulation number
                parts = scenario_part.split('_')
                if parts:
                    try:
                        sim_number = int(parts[-1])
                        extracted_data['simulation_number'] = sim_number
                    except ValueError:
                        extracted_data['simulation_number'] = None
            else:
                extracted_data['scenario'] = 'unknown'
                extracted_data['simulation_number'] = None
            
            # For temporal mode, extract year values
            if analysis_mode == 'temporal':
                for var_name in variables_to_extract:
                    for col in columns_to_extract:
                        var_row = df[df['Variables'] == var_name]
                        if not var_row.empty:
                            try:
                                value = float(str(var_row.iloc[0][col]).replace(',', '.'))
                            except (ValueError, AttributeError):
                                value = str(var_row.iloc[0][col])
                            
                            # Store with combined key for temporal data
                            if var_name in [y_variable, y_variable_2]:
                                year = col.replace('values_', '')
                                extracted_data[f"{var_name}_{year}"] = value
                            else:
                                # Parameters are the same across years
                                extracted_data[var_name] = value
                        else:
                            # Variable not found in this file
                            if var_name in [y_variable, y_variable_2]:
                                year = col.replace('values_', '')
                                print(f"Warning in {file_path.name}: Variable '{var_name}' not found for column '{col}'")
                                extracted_data[f"{var_name}_{year}"] = None
            else:
                # Single year mode: extract from specific time column
                for var_name in variables_to_extract:
                    var_row = df[df['Variables'] == var_name]
                    if not var_row.empty:
                        try:
                            value = float(str(var_row.iloc[0][time_column]).replace(',', '.'))
                        except (ValueError, AttributeError):
                            value = str(var_row.iloc[0][time_column])
                        
                        extracted_data[var_name] = value
            
            # Check if we have all required data
            required_keys = []
            if color_parameter:
                required_keys.append(color_parameter)
            if linestyle_parameter:
                required_keys.append(linestyle_parameter)
            if marker_parameter:
                required_keys.append(marker_parameter)
            if analysis_mode == 'temporal':
                for col in columns_to_extract:
                    year = col.replace('values_', '')
                    required_keys.append(f"{y_variable}_{year}")
                    if y_operation and y_variable_2:
                        required_keys.append(f"{y_variable_2}_{year}")
            else:
                required_keys.extend([x_variable, y_variable])
                if y_operation and y_variable_2:
                    required_keys.append(y_variable_2)
            
            if all(key in extracted_data for key in required_keys):
                data.append(extracted_data)
            else:
                missing = [key for key in required_keys if key not in extracted_data]
                print(f"Missing data in {file_path.name}: {missing}")
                
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
    
    return data

# Read AMS files
data_ams = extract_data_from_files(fulltemplate_files_ams, 'AMS')

# Read AME files if comparison is requested
if compare_scenarios:
    data_ame = extract_data_from_files(fulltemplate_files_ame, 'AME')

# Convert to DataFrame
df_ams = pd.DataFrame(data_ams)
df_ame = pd.DataFrame(data_ame) if compare_scenarios else None

print(f"\nFound {len(df_ams)} AMS simulations")
if compare_scenarios:
    print(f"Found {len(df_ame)} AME simulations")

# Apply operation first if specified (BEFORE scenario comparison)
if y_operation and y_variable_2:
    if analysis_mode == 'temporal':
        # For temporal mode, apply operation across years for AMS
        for col in temporal_columns:
            year = col.replace('values_', '')
            y1_col = f"{y_variable}_{year}"
            y2_col = f"{y_variable_2}_{year}"
            result_col = f"result_{year}"
            
            if y_operation == 'difference':
                df_ams[result_col] = df_ams[y2_col] - df_ams[y1_col]
            elif y_operation == 'ratio':
                df_ams[result_col] = df_ams[y2_col] / df_ams[y1_col]
            elif y_operation == 'percentage_change':
                df_ams[result_col] = (df_ams[y2_col] - df_ams[y1_col]) / df_ams[y1_col] * 100
        
        # Apply operation for AME if comparing scenarios
        if compare_scenarios and df_ame is not None:
            for col in temporal_columns:
                year = col.replace('values_', '')
                y1_col = f"{y_variable}_{year}"
                y2_col = f"{y_variable_2}_{year}"
                result_col = f"result_{year}"
                
                if y_operation == 'difference':
                    df_ame[result_col] = df_ame[y2_col] - df_ame[y1_col]
                elif y_operation == 'ratio':
                    df_ame[result_col] = df_ame[y2_col] / df_ame[y1_col]
                elif y_operation == 'percentage_change':
                    df_ame[result_col] = (df_ame[y2_col] - df_ame[y1_col]) / df_ame[y1_col] * 100
        
        # Update y_variable to use the result
        original_y_variable = y_variable
        y_variable = 'result'
    else:
        # For single year mode, apply operation for AMS
        if y_operation == 'difference':
            df_ams['y_result'] = df_ams[y_variable_2] - df_ams[y_variable]
            operation_label = f"{y_variable_2} - {y_variable}"
        elif y_operation == 'ratio':
            df_ams['y_result'] = df_ams[y_variable_2] / df_ams[y_variable]
            operation_label = f"{y_variable_2} / {y_variable}"
        elif y_operation == 'percentage_change':
            df_ams['y_result'] = (df_ams[y_variable_2] - df_ams[y_variable]) / df_ams[y_variable] * 100
            operation_label = f"({y_variable_2} - {y_variable}) / {y_variable} × 100"
        
        # Apply operation for AME if comparing scenarios
        if compare_scenarios and df_ame is not None:
            if y_operation == 'difference':
                df_ame['y_result'] = df_ame[y_variable_2] - df_ame[y_variable]
            elif y_operation == 'ratio':
                df_ame['y_result'] = df_ame[y_variable_2] / df_ame[y_variable]
            elif y_operation == 'percentage_change':
                df_ame['y_result'] = (df_ame[y_variable_2] - df_ame[y_variable]) / df_ame[y_variable] * 100
        
        # Replace y_variable with result
        original_y_variable = y_variable
        y_variable = 'y_result'

# Apply Y-axis format transformation if requested
if y_axis_format == 'percentage':
    if analysis_mode == 'temporal':
        # Convert all temporal columns to percentage
        for col in temporal_columns:
            year = col.replace('values_', '')
            y_col = f"{y_variable}_{year}"
            if y_col in df_ams.columns:
                df_ams[y_col] = df_ams[y_col] * 100
        if compare_scenarios and df_ame is not None:
            for col in temporal_columns:
                year = col.replace('values_', '')
                y_col = f"{y_variable}_{year}"
                if y_col in df_ame.columns:
                    df_ame[y_col] = df_ame[y_col] * 100
    else:
        # Convert single year column to percentage
        if y_variable in df_ams.columns:
            df_ams[y_variable] = df_ams[y_variable] * 100
        if compare_scenarios and df_ame is not None and y_variable in df_ame.columns:
            df_ame[y_variable] = df_ame[y_variable] * 100

# Handle scenario comparison if requested (AFTER operation)
if compare_scenarios and df_ame is not None:
    # Merge AMS and AME data on parameters only (not simulation_number, as numbers may differ)
    merge_keys = []
    if color_parameter:
        merge_keys.append(color_parameter)
    if linestyle_parameter:
        merge_keys.append(linestyle_parameter)
    if marker_parameter:
        merge_keys.append(marker_parameter)
    
    if analysis_mode == 'temporal':
        # For temporal mode, calculate differences for each year
        df_merged = pd.merge(df_ams, df_ame, on=merge_keys, suffixes=('_AMS', '_AME'))
        
        for col in temporal_columns:
            year = col.replace('values_', '')
            ams_col = f"{y_variable}_{year}_AMS"
            ame_col = f"{y_variable}_{year}_AME"
            result_col = f"{y_variable}_{year}"
            
            if compare_scenarios == 'ams_ame_difference':
                df_merged[result_col] = df_merged[ams_col] - df_merged[ame_col]
            elif compare_scenarios == 'ams_ame_percentage':
                df_merged[result_col] = (df_merged[ams_col] - df_merged[ame_col]) / df_merged[ame_col] * 100
        
        df_results = df_merged
    else:
        # For single year mode
        df_merged = pd.merge(df_ams, df_ame, on=merge_keys, suffixes=('_AMS', '_AME'))
        
        ams_col = f"{y_variable}_AMS"
        ame_col = f"{y_variable}_AME"
        
        if compare_scenarios == 'ams_ame_difference':
            df_merged[y_variable] = df_merged[ams_col] - df_merged[ame_col]
        elif compare_scenarios == 'ams_ame_percentage':
            df_merged[y_variable] = (df_merged[ams_col] - df_merged[ame_col]) / df_merged[ame_col] * 100
        
        df_results = df_merged
else:
    df_results = df_ams

print("\nData summary:")
print(df_results.head())

# Note: Operation has already been applied above before scenario comparison
# No need to apply operation again here

# Get unique values for color, linestyle, and marker parameters
color_values = sorted(df_results[color_parameter].unique()) if color_parameter else [None]
linestyle_values = sorted(df_results[linestyle_parameter].unique()) if linestyle_parameter else [None]
marker_values = sorted(df_results[marker_parameter].unique()) if marker_parameter else [None]

if color_parameter:
    print(f"\n{color_parameter} values: {color_values}")
if linestyle_parameter:
    print(f"{linestyle_parameter} values: {linestyle_values}")
if marker_parameter:
    print(f"{marker_parameter} values: {marker_values}")

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Define color palette (automatically adjusted to number of unique values)
color_palette = ['#0066CC', '#CC0066', '#00CC66', '#FF6600', '#9933CC', '#CCCC00']
colors_dict = {val: color_palette[i % len(color_palette)] for i, val in enumerate(color_values)}

# Define line styles (automatically adjusted to number of unique values)
linestyle_options = ['-', (0, (5, 2)), (0, (1, 1)), (0, (3, 5, 1, 5)), (0, (5, 1))]
linestyles_dict = {val: linestyle_options[i % len(linestyle_options)] for i, val in enumerate(linestyle_values)}

# Define marker styles
marker_options = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'P', 'X']
if marker_parameter:
    markers_dict = {val: marker_options[i % len(marker_options)] for i, val in enumerate(marker_values)}
else:
    # If no marker_parameter, use different markers for linestyle_parameter
    markers_dict = {val: marker_options[i % len(marker_options)] for i, val in enumerate(linestyle_values)}

print(f"\nColors mapping: {colors_dict}")
print(f"Linestyles mapping: {linestyles_dict}")
if marker_parameter:
    print(f"Markers mapping ({marker_parameter}): {markers_dict}")
else:
    print(f"Markers mapping ({linestyle_parameter}): {markers_dict}")

# Save data to CSV if requested
if save_csv:
    # Generate CSV filename if not provided
    if csv_filename is None:
        if analysis_mode == 'temporal':
            if compare_scenarios == 'ams_ame_difference':
                csv_filename = f"data_temporal_AMS_AME_diff_{y_variable}.csv"
            elif compare_scenarios == 'ams_ame_percentage':
                csv_filename = f"data_temporal_AMS_AME_pct_{y_variable}.csv"
            elif y_operation and y_variable_2:
                csv_filename = f"data_temporal_{y_operation}_{original_y_variable}_{y_variable_2}.csv"
            else:
                csv_filename = f"data_temporal_{y_variable}.csv"
        else:
            if compare_scenarios == 'ams_ame_difference':
                csv_filename = f"data_AMS_AME_diff_{y_variable}_vs_{x_variable}.csv"
            elif compare_scenarios == 'ams_ame_percentage':
                csv_filename = f"data_AMS_AME_pct_{y_variable}_vs_{x_variable}.csv"
            elif y_operation and y_variable_2:
                csv_filename = f"data_{y_operation}_{original_y_variable}_{y_variable_2}_vs_{x_variable}.csv"
            else:
                csv_filename = f"data_{y_variable}_vs_{x_variable}.csv"
    
    # Save to CSV
    csv_path = base_dir / csv_filename
    df_results.to_csv(csv_path, index=False, sep=';', decimal=',')
    print(f"\nData saved to CSV: {csv_path}")
    print(f"Number of rows: {len(df_results)}")
    print(f"Columns: {list(df_results.columns)}")

# Plot each combination
for color_val in color_values:
    for linestyle_val in linestyle_values:
        for marker_val in marker_values:
            # Filter data for this combination
            if marker_parameter:
                mask = ((df_results[color_parameter] == color_val) & 
                       (df_results[linestyle_parameter] == linestyle_val) &
                       (df_results[marker_parameter] == marker_val))
            else:
                mask = (df_results[color_parameter] == color_val) & (df_results[linestyle_parameter] == linestyle_val)
            
            subset = df_results[mask]
            
            if len(subset) > 0:
                # Create descriptive label
                if marker_parameter:
                    label = f'{color_parameter}={color_val}, {linestyle_parameter}={linestyle_val}, {marker_parameter}={marker_val}'
                else:
                    label = f'{color_parameter}={color_val}, {linestyle_parameter}={linestyle_val}'
            
            if analysis_mode == 'temporal':
                # For temporal analysis, plot years on X-axis
                years = [col.replace('values_', '') for col in temporal_columns]
                y_values = []
                
                for col in temporal_columns:
                    year = col.replace('values_', '')
                    y_col = f"{y_variable}_{year}"
                    # Average across all simulations with same parameters
                    y_values.append(subset[y_col].mean())
                
                # Plot temporal evolution
                marker = markers_dict.get(marker_val if marker_parameter else linestyle_val, 'o')
                ax.plot(years, 
                       y_values,
                       color=colors_dict.get(color_val, 'gray'),
                       linestyle=linestyles_dict.get(linestyle_val, '-'),
                       marker=marker,
                       linewidth=2.5,
                       markersize=8,
                       label=label,
                       alpha=0.8)
            else:
                # For single year analysis, plot x_variable vs y_variable
                subset = subset.sort_values(x_variable)
                marker = markers_dict.get(marker_val if marker_parameter else linestyle_val, 'o')
                
                ax.plot(subset[x_variable], 
                       subset[y_variable],
                       color=colors_dict.get(color_val, 'gray'),
                       linestyle=linestyles_dict.get(linestyle_val, '-'),
                       marker=marker,
                       linewidth=2.5,
                       markersize=8,
                       label=label,
                       alpha=0.8)

# Add ThreeME data if requested (in temporal mode with scenario comparison)
if (show_threeme and analysis_mode == 'temporal' 
    and compare_scenarios in ['ams_ame_percentage', 'ams_ame_difference']):
    # Extract years that are in both temporal_columns and threeme_data
    threeme_years = []
    threeme_values = []
    
    for col in temporal_columns:
        year = col.replace('values_', '')
        if year in threeme_data and year != '2018':  # Exclude 2018 as it's the base year
            threeme_years.append(year)
            threeme_values.append(threeme_data[year])
    
    if threeme_years:
        ax.plot(threeme_years, 
               threeme_values,
               color='black',
               linestyle='-',
               marker='D',
               linewidth=3,
               markersize=10,
               label='ThreeME',
               alpha=0.9,
               zorder=100)  # High zorder to ensure it's on top

# Add a horizontal line at y=1 for reference (only if y_variable is scal_markup)
if 'scal_markup' in str(y_variable) and not y_operation:
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Generate automatic labels if not provided
if plot_title is None:
    if analysis_mode == 'temporal':
        if compare_scenarios == 'ams_ame_difference':
            plot_title = f'Différence AMS - AME : {y_variable}'
        elif compare_scenarios == 'ams_ame_percentage':
            plot_title = f'Différence relative AMS vs AME (%) : {y_variable}'
        elif y_operation and y_variable_2:
            plot_title = f'Évolution temporelle : {y_operation}({original_y_variable}, {y_variable_2})'
        else:
            plot_title = f'Évolution temporelle de {original_y_variable if y_operation else y_variable}'
    else:
        if compare_scenarios == 'ams_ame_difference':
            plot_title = f'Différence AMS - AME : {y_variable} en fonction de {x_variable} ({time_column.replace("values_", "")})'
        elif compare_scenarios == 'ams_ame_percentage':
            plot_title = f'Différence relative AMS vs AME (%) : {y_variable} en fonction de {x_variable} ({time_column.replace("values_", "")})'
        elif y_operation and y_variable_2:
            plot_title = f'{operation_label} en fonction de {x_variable} ({time_column.replace("values_", "")})'
        else:
            plot_title = f'{y_variable} en fonction de {x_variable} ({time_column.replace("values_", "")})'

if x_label is None:
    x_label = 'Année' if analysis_mode == 'temporal' else x_variable.replace('_', ' ').title()

if y_label is None:
    if compare_scenarios == 'ams_ame_difference':
        base_label = f'{y_variable} (AMS - AME)'
    elif compare_scenarios == 'ams_ame_percentage':
        base_label = f'{y_variable} - Différence relative (%)'
    elif y_operation and y_variable_2:
        if y_operation == 'difference':
            base_label = f'{y_variable_2} - {original_y_variable if y_operation else y_variable}'
        elif y_operation == 'ratio':
            base_label = f'{y_variable_2} / {original_y_variable if y_operation else y_variable}'
        elif y_operation == 'percentage_change':
            base_label = f'Variation % ({y_variable_2}/{original_y_variable if y_operation else y_variable})'
    else:
        base_label = (original_y_variable if 'original_y_variable' in locals() else y_variable).replace('_', ' ').title()
    
    # Add percentage suffix if format is percentage
    if y_axis_format == 'percentage':
        y_label = f'{base_label} (%)'
    else:
        y_label = base_label

# Customize plot
ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
ax.set_title(plot_title, fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)

# Create custom legend with two sections
# First, collect handles and labels
handles, labels = ax.get_legend_handles_labels()

# Create legend with title - placed outside the plot on the right
if marker_parameter:
    legend_title = f'Paramètres de simulation\n({color_parameter} par couleur,\n{linestyle_parameter} par ligne,\n{marker_parameter} par marqueur)'
else:
    legend_title = f'Paramètres de simulation\n({color_parameter} par couleur, {linestyle_parameter} par ligne)'

legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9, 
                   title=legend_title,
                   title_fontsize=9)

# Add text annotation for reference line (only if y_variable is scal_markup)
if 'scal_markup' in str(y_variable) and not y_operation:
    ax.text(0.02, 0.98, 'Ligne de référence: scal_markup = 1', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save the figure
if analysis_mode == 'temporal':
    if compare_scenarios == 'ams_ame_difference':
        output_filename = f"plot_temporal_AMS_AME_diff_{y_variable}.png"
    elif compare_scenarios == 'ams_ame_percentage':
        output_filename = f"plot_temporal_AMS_AME_pct_{y_variable}.png"
    elif y_operation and y_variable_2:
        output_filename = f"plot_temporal_{y_operation}_{original_y_variable}_{y_variable_2}.png"
    else:
        output_filename = f"plot_temporal_{y_variable}.png"
else:
    if compare_scenarios == 'ams_ame_difference':
        output_filename = f"plot_AMS_AME_diff_{y_variable}_vs_{x_variable}.png"
    elif compare_scenarios == 'ams_ame_percentage':
        output_filename = f"plot_AMS_AME_pct_{y_variable}_vs_{x_variable}.png"
    elif y_operation and y_variable_2:
        output_filename = f"plot_{y_operation}_{original_y_variable}_{y_variable_2}_vs_{x_variable}.png"
    else:
        output_filename = f"plot_{y_variable}_vs_{x_variable}.png"

output_path = base_dir / output_filename
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Show the plot
plt.show()
