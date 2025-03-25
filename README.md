# datasets

## Deployment and Production Setup

### Prerequisites

- Python 3.7 or higher
- Git
- GitHub account and access token (optional but recommended for higher API rate limits)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/fortunestoldco/datasets.git
cd datasets
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

1. Set up environment variables:

Create a `.env` file in the root directory of the project and add the following variables:

```
GITHUB_TOKEN=your_github_token_here
```

Replace `your_github_token_here` with your actual GitHub token.

### Running the Application

To run the application, use the following command:

```bash
python app.py
```

### Testing

To run the unit tests and integration tests, use the following command:

```bash
pytest
```

### Continuous Integration (CI)

This repository uses GitHub Actions for continuous integration. The CI pipeline is defined in the `.github/workflows/ci.yml` file. It runs the tests and checks for code quality on every push and pull request.

### Deployment

To deploy the application, follow these steps:

1. Set up a server or cloud instance (e.g., AWS, Azure, GCP).
2. Install the required software (Python, Git, etc.) on the server.
3. Clone the repository and set up the environment as described in the Installation section.
4. Configure a process manager (e.g., `systemd`, `supervisord`) to run the application as a service.
5. Set up a reverse proxy (e.g., Nginx) to forward requests to the application.

### Monitoring and Logging

To monitor the application's performance and logs, consider using tools like:

- Prometheus and Grafana for monitoring
- ELK stack (Elasticsearch, Logstash, Kibana) for logging and visualization
