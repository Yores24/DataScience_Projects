import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')

# Ensure directory exists
def ensure_directory(directory_path):
    """
    Check if a directory exists, and create it if not.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist. Creating it.")
        os.makedirs(directory_path, exist_ok=True)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to engineer features on job title
def engineer_job_title_features(df):
    def new_title(title):
        if 'data scientist' in title.lower():
            return 'data scientist'
        elif 'analyst' in title.lower() or 'analysis' in title.lower():
            return 'analyst'
        elif 'manager' in title.lower():
            return 'manager'
        elif 'engineer' in title.lower():
            return 'data engineer'
        elif 'machine learning' in title.lower():
            return 'mle'
        elif 'director' in title.lower():
            return 'director'
        else:
            return 'na'

    def senior_func(title):
        if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
            return 'sr'
        elif 'jr' in title.lower() or 'jr.' in title.lower():
            return 'jr'
        else:
            return 'na'

    df['job_simp'] = df['Job Title'].apply(new_title)
    df['job_senior'] = df['Job Title'].apply(senior_func)
    return df

# Function to clean location and competitors
def clean_location_and_competitors(df):
    df['Job_state'] = df['Job_state'].apply(lambda x: 'LA' if 'Los Angeles' in x else x.strip())
    df['comp_count'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)
    return df

# Function to calculate description length
def calculate_description_length(df):
    df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
    return df

# Function to clean salary columns
def clean_salary_columns(df):
    df['min_salary'] = df.apply(lambda x: x.min_salary * 2 if x.Hour == 1 else x.min_salary, axis=1)
    df['max_salary'] = df.apply(lambda x: x.max_salary * 2 if x.Hour == 1 else x.max_salary, axis=1)
    return df

# Function to clean job description text
def clean_job_description(df):
    df['Job Description'] = df['Job Description'].apply(lambda x: x.replace("\r\n", ' '))
    return df

# Function to save plots
def save_plots(df, output_folder):
    ensure_directory(output_folder)

    # Histogram plots
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    df['Rating'].hist()
    plt.savefig(os.path.join(output_folder, 'rating_hist.png'))
    plt.subplot(1, 3, 2)
    df['avg_salary'].hist()
    plt.savefig(os.path.join(output_folder, 'avg_salary_hist.png'))
    plt.subplot(1, 3, 3)
    df['Age'].hist()
    plt.savefig(os.path.join(output_folder, 'age_hist.png'))
    plt.close()

    # Box plots
    df.boxplot(column=['Age', 'avg_salary'])
    plt.savefig(os.path.join(output_folder, 'age_avg_salary_boxplot.png'))
    plt.close()

    df.boxplot('Rating')
    plt.savefig(os.path.join(output_folder, 'rating_boxplot.png'))
    plt.close()

    # Heatmap
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True)
    plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
    plt.close()

# Function to create pivot tables
def save_pivot_tables(df, output_folder):
    ensure_directory(output_folder)
    pivot_table_job = pd.pivot_table(df, index='job_simp', values='avg_salary')
    pivot_table_job.to_csv(os.path.join(output_folder, 'pivot_job_simp.csv'))

    pivot_table_state = pd.pivot_table(df, index='Job_state', values='avg_salary').sort_values('avg_salary', ascending=False)
    pivot_table_state.to_csv(os.path.join(output_folder, 'pivot_job_state.csv'))

# Function to generate and save a word cloud
def save_wordcloud(df, output_folder):
    ensure_directory(output_folder)
    words = " ".join(df['Job Description'])
    
    def punctuation_stop(text):
        filtered = []
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        for w in word_tokens:
            if w not in stop_words and w.isalpha():
                filtered.append(w.lower())
        return filtered

    words_filter = punctuation_stop(words)
    text = " ".join(words_filter)

    wc = WordCloud(background_color='white', random_state=1, stopwords=STOPWORDS, max_words=2000, width=800, height=1500)
    wc.generate(text)

    plt.figure(figsize=[10, 10])
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, 'wordcloud.png'))
    plt.close()

# Main function
def main(input_file, output_cleaned_file, plot_folder, pivot_folder, wordcloud_folder):
    # Load data
    df = load_data(input_file)

    # Clean and process data
    df = engineer_job_title_features(df)
    df = clean_location_and_competitors(df)
    df = calculate_description_length(df)
    df = clean_salary_columns(df)
    df = clean_job_description(df)

    # Save cleaned data
    ensure_directory(os.path.dirname(output_cleaned_file))
    df.to_csv(output_cleaned_file, index=False)

    # Generate plots
    save_plots(df, plot_folder)

    # Generate pivot tables
    save_pivot_tables(df, pivot_folder)

    # Generate word cloud
    save_wordcloud(df, wordcloud_folder)

if __name__ == "__main__":
    main(
        input_file="data/processed/cleaned_salary.csv",
        output_cleaned_file="data/processed/eda_data.csv",
        plot_folder="reports/figures/plots",
        pivot_folder="reports/figures/pivot_tables",
        wordcloud_folder="reports/figures/wordclouds"
    )
