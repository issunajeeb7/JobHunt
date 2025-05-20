# JobHunt: Smart Job Discovery & Alerts

JobHunt is an intelligent job monitoring and recommendation system for data science jobs in India, powered by web scraping, unsupervised machine learning, and a modern Streamlit UI.

## Features
- **Daily scraping** of job listings from [Karkidi.com](https://www.karkidi.com)
- **Automatic clustering** of jobs based on required skills (unsupervised ML)
- **Skill-based job search**: Find jobs matching your interests
- **On-demand refresh**: Update job listings from the Streamlit app

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd JobHunt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Scrape and Cluster Jobs (Manual)
```bash
python scrape.py
```
This will fetch the latest jobs and save them to `data/jobs.csv`.

### 2. Run the Streamlit App
```bash
streamlit run app.py
```
- Use the "🔄 Refresh Job Listings Now" button to update jobs and clusters from the app UI.
- Search for jobs by your skills and sign up for notifications.

### 3. Automate Daily Scraping (Optional)
- **Windows:** Use Task Scheduler to run `python scrape.py` daily.
- **Linux/Mac:** Add a cron job:
  ```
  0 6 * * * cd /path/to/JobHunt && /path/to/python scrape.py
  ```

## Troubleshooting
- **PermissionError:** If you see a `Permission denied: 'data/jobs.csv'` error, make sure the file is not open in Excel or any other program, and that you have write permissions.
- **Dependencies:** If you have issues with packages, try upgrading pip and reinstalling requirements.

## Project Structure
```
JobHunt/
├── app.py
├── scrape.py
├── model.py
├── requirements.txt
├── README.md
└── data/
    ├── jobs.csv
    └── jobs_clustered.csv
```

## License
MIT 
