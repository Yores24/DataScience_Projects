import os
import pandas as pd
from logger import CustomLogger  # Assuming the logger class is in a file named `logger_file.py`

# Function to load data
def load_data(file_path, logger):
    logger.log_message(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

# Function to clean the salary estimate column
def clean_salary_estimate(df, logger):
    logger.log_message("Cleaning Salary Estimate column")
    df = df[df['Salary Estimate'] != '-1']
    df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.split()[0])
    df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.replace('K', '').replace('$', ''))
    df['Hour'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
    df['Employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)
    return df

# Function to handle employer-provided salary parsing
def parse_employer_salary(df, df_copy, logger):
    logger.log_message("Parsing employer-provided salary data")
    new = []
    for i in range(len(df)):
        if df['Employer_provided'].iloc[i] == 1:
            new.append(df_copy['Salary Estimate'].iloc[i])
        else:
            new.append(df['Salary Estimate'].iloc[i])
    df['Salary Estimate'] = new
    return df

# Function to extract min, max, and average salary
def extract_salary(df, logger):
    logger.log_message("Extracting min, max, and average salary")
    df['Salary Estimate'] = df['Salary Estimate'].apply(lambda x: x.replace('(Employer', '').replace('Employer Provided Salary:', '').replace('Per Hour', ''))
    df['min_salary'] = df['Salary Estimate'].apply(lambda x: int(x.split('-')[0]))
    df['max_salary'] = df['Salary Estimate'].apply(lambda x: int(x.split('-')[1]))
    df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
    return df

# Function to clean company name
def clean_company_name(df, logger):
    logger.log_message("Cleaning company names")
    df['Company Name'] = df['Company Name'].apply(lambda x: x.split('\n')[0])
    return df

# Function to extract job state
def extract_job_state(df, logger):
    logger.log_message("Extracting job state from location")
    df['Job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
    return df

# Function to check if location and headquarters are in the same place
def check_same_location(df, logger):
    logger.log_message("Checking if job location and headquarters are in the same state")
    df['Same_state'] = df.apply(lambda x: 1 if x['Location'] == x['Headquarters'] else 0, axis=1)
    return df

# Function to calculate company age
def calculate_company_age(df, current_year, logger):
    logger.log_message("Calculating company age")
    df['Age'] = df['Founded'].apply(lambda x: x if x < 1 else current_year - x)
    return df

# Function to parse job description for specific skills
def parse_job_description(df, logger):
    logger.log_message("Parsing job descriptions for skills")
    df['python_pres'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
    df['r_pres'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() else 0)
    df['spark_pres'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
    df['aws_pres'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
    df['excel_pres'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
    return df

# Function to check if the directory exists, and if not, create it
def check_and_create_directory(directory_path, logger):
    if not os.path.exists(directory_path):
        logger.log_message(f"Creating directory: {directory_path}")
        os.makedirs(directory_path)

# Main function to orchestrate the data processing
def main(file_path, output_path):
    # Create logger
    logger = CustomLogger(logger_name="GlassdoorProcessor")

    try:
        # Load data
        df = load_data(file_path, logger)
        df_copy = df.copy()

        # Clean and parse data
        df = clean_salary_estimate(df, logger)
        df = parse_employer_salary(df, df_copy, logger)
        df = extract_salary(df, logger)
        df = clean_company_name(df, logger)
        df = extract_job_state(df, logger)
        df = check_same_location(df, logger)
        df = calculate_company_age(df, current_year=2023, logger=logger)
        df = parse_job_description(df, logger)

        # Drop unnecessary column and save final data
        if 'Unnamed: 0' in df.columns:
            logger.log_message("Dropping unnecessary 'Unnamed: 0' column")
            df = df.drop(['Unnamed: 0'], axis=1)

        # Check if the output directory exists and create it if not
        output_directory = os.path.dirname(output_path)
        check_and_create_directory(output_directory, logger)

        logger.log_message(f"Saving cleaned data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.log_message("Data processing completed successfully!")

    except Exception as e:
        logger.log_message(f"An error occurred: {e}")

# Run the main function
if __name__ == "__main__":
    main("data/raw/glassdoor_jobs.csv", "data/processed/cleaned_salary.csv")
