# GitHub Publishing Review Tasks for Sentrl Project

This document outlines the tasks required to prepare the Sentrl project (including `sentrl_demo_webapp` and `sentrl_agentic_platform_tools`) for public release on GitHub with a GNU AGPLv3 license, ensuring PII protection and good open-source practices.

## Phase 1: Auditing and PII/Secrets Identification

### Task 1: Overall Project Structure and Initial Setup Review
- **1.1.** Confirm understanding of the project's dual components: `sentrl_demo_webapp` (Django/React) and `sentrl_agentic_platform_tools` (Python AI/ML tools).
- **1.2.** Document the high-level directory structure of the entire workspace.

### Task 2: Audit `sentrl_demo_webapp/`
- **2.1. `.gitignore` Review & Enhancement:**
    - Ensure `node_modules/` (especially for `sentrl_demo_webapp/chat_simulation_app/`) is ignored.
    - Ensure `*.DS_Store` is ignored.
    - Confirm `certs/` is appropriately handled (ignored if sensitive, or contents vetted).
    - Confirm `db.sqlite3`, `media/`, `staticfiles/`, `ml_models/` are correctly ignored.
- **2.2. `sentrl_project/settings.py` Audit (CRITICAL for PII/Secrets):**
    - Identify all sensitive values: `SECRET_KEY`, `DATABASES` credentials (if not SQLite or if SQLite path is sensitive), API keys (e.g., Ollama), email server settings, any other third-party service credentials.
    - Plan for all secrets to be loaded from environment variables.
- **2.3. Database File (`db.sqlite3`):**
    - Confirm it's in `.gitignore` and will **not** be committed.
- **2.4. Certificates (`certs/` directories):**
    - Review contents of `sentrl_demo_webapp/certs/` and `sentrl_demo_webapp/sentrl_project/certs/`.
    - Determine sensitivity. If sensitive, ensure they are gitignored and document for users if they need to generate their own. Clarify if one of these directories is redundant.
- **2.5. Media Files (`media/` directories):**
    - Review contents of `sentrl_demo_webapp/media/` and `sentrl_demo_webapp/sentrl_project/media/`.
    - Confirm these are for user-uploaded content (potential PII) and are correctly gitignored. Clarify if one is redundant.
- **2.6. Machine Learning Models (`ml_models/`):**
    - Confirm `ml_models/` is in `.gitignore`.
- **2.7. Scripts (`scripts/download_model.py`):**
    - Review for hardcoded URLs, API keys, or other sensitive information. Plan to make these configurable or use environment variables.
- **2.8. Application Code Audit (`landing/`, `chat_simulation_app/`, `sentrl_project/` excluding `settings.py`):**
    - Review Python files (`views.py`, `models.py`, `forms.py`, `admin.py`, etc.) for embedded PII, hardcoded secrets, sensitive comments, or local paths.
    - Review JavaScript/React files (`chat_simulation_app/src/`, `convert_conversations.js`, `chat_experience_demo.js`, `landing/static/`) for embedded PII, hardcoded API endpoints, keys, or secrets.
- **2.9. Configuration Files (webpack, babel, tailwind, postcss):**
    - Briefly review for any accidental inclusion of sensitive paths or keys (less common but possible).
- **2.10. Redundancy and Non-Standard Items:**
    - Identify any `.DS_Store` files for removal via `.gitignore`.
    - Check for any other local development artifacts that shouldn't be committed.
    - Note potential redundancies (e.g., multiple `certs/` or `media/` dirs).

### Task 3: Audit `sentrl_agentic_platform_tools/`
- **3.1. General Audit Strategy:**
    - For each subdirectory, look for configuration files (e.g., `.json`, `.yaml`, `.ini`, `.env-template`), Python scripts, Jupyter notebooks (`.ipynb`), shell scripts (`.sh`), and data files (especially `.json`, `.csv`, text files).
- **3.2. Subdirectory-Specific Audits for PII/Secrets/Config:**
    - **`action_instruction_tuning_formatting/`**:
        - Review `current-state-data_instruct.json` (353KB) for any embedded PII from instruction prompts or state data.
        - Review Python scripts (`state-desc-instruct-alpaca-jsonl_v2.1.py`, `OAI-chat-format_v0.0.py`, `state-desc-instruction_v0.1.py`) for hardcoded paths, API keys, or PII handling.
    - **`fine_tuned_models/`**:
        - **`smb_needs_discover_chatbot/`** & **`travel_assistant_chabot/`** (and any similar model directories):
            - Assess model files (e.g., in `llama-3.2-3b-instruct/`) for size (consider Git LFS or download instructions if large) and potential for PII reproduction.
            - Review Jupyter Notebooks (e.g., `Unsloth_Llama3_2_(1B_and_3B)_Conversational.ipynb`) for PII in code or output cells. **Action: Clear sensitive outputs before committing or export to a script/markdown.**
            - Check any accompanying configuration or metadata files for PII.
        - Determine strategy: Commit if models are small, essential, and PII-free. Otherwise, add to `.gitignore` and provide download/training instructions in README.
    - **`action_qdrant_loader/`**:
        - Identify and review main Python scripts (if any, not visible in initial listing beyond `Embeddings/`) for Qdrant connection URLs/API keys (e.g., if using Qdrant Cloud) and paths to data sources. Ensure secrets are loaded from environment variables.
        - Inspect contents of `Embeddings/` directory for data derived from PII.
    - **`action_datalabeling/`**:
        - Review `ground-truth-form_v1.8.py` (57KB) for hardcoded paths to datasets, data storage connections, or direct PII handling/display logic.
    - **`synthetic_data_generators/`**:
        - Review `start_script.sh` for hardcoded paths or credentials.
        - **`synthetic_data_gen/`** (Python package):
            - Review `README.md`, `ARCHITECTURE.md` for sensitive internal details.
            - Examine core generation scripts (within inner `synthetic_data_gen/`) and `examples/` for PII in seed data, generation logic that might replicate PII, or use of API keys.
        - **`synthetic_data_test/`**, **`synthetic_data/`** (potential output), **`task manager/`**, **`needs discovery/`**, **`travel assistant/`**: Inspect contents for PII in scripts or data.
        - Clarify purpose of **`chat_simulation_app/`** subdirectory: If it's a copy of the one in `sentrl_demo_webapp`, it might be redundant. Check for PII.
    - **`action_auto_labelling/`**:
        - Review Python scripts (`llama3.2-vision-annotation_v1.0.py`, `llama3.2-vision-annotation_v0.py`, `chatgpt4o-annotation-json_v0.py`) for:
            - API key management for LLMs (Llama 3.2, ChatGPT-4o) – ensure keys are loaded from environment variables, not hardcoded.
            - Hardcoded paths for data input.
            - Logging or storage of data containing PII from inputs or LLM responses.
    - **`user_action_logger/` (HIGH RISK - CRITICAL REVIEW NEEDED):**
        - Review Python scripts (`action_logger_v0.1.py`, `mouse-capture-action-logger-v0.0.py`, `datalogger_json_oop_v0.0.py`, `concat_actions_train.py`, `data-processing-fusion.py`) to understand:
            - Specific user actions logged (keystrokes, mouse events, window titles, application names, clipboard content, form inputs, URLs visited, etc.).
            - Data storage mechanisms (local files, database configurations) – check for hardcoded paths or credentials.
        - Review scripts in **`Data Transforms/`** for PII processing.
        - **Action: Remove or thoroughly anonymize any example log files containing your PII or specific prompts.**
        - **Strategy for release:** Ensure extreme clarity to users about what is logged. Provide strong warnings about PII. If possible, implement PII filtering/anonymization (though this is difficult). The tool should primarily be for users to log *their own* actions on *their own* systems.
    - **`action_mongodb_loader/`**:
        - Review Python scripts (`raw-actions_v0.0.py`, `actions-mongodb_v0.1.py`) for MongoDB connection string handling (server address, port, username, password, database name). Ensure these are loaded from environment variables.
        - Review `load-mongo-format.json` for schema details.
        - **Review `sample_actions.json` (192KB) thoroughly for PII. Anonymize or remove if it contains your PII.**
- **3.3. General PII/Secrets in Code/Scripts:**
    - Review Python scripts and any notebooks for hardcoded API keys, passwords, local file paths, or embedded PII.
- **3.4. Redundancy and Non-Standard Items:**
    - Identify any `.DS_Store` files for removal via `.gitignore`.
    - Check for compiled files, caches, or local outputs that shouldn't be committed.

## Phase 2: Remediation and Preparation for Publishing

### Task 4: Root Directory and Overall Project Finalization
- **4.1. Root `.gitignore`:**
    - Create or consolidate into a single root `.gitignore` file. This file should cover:
        - Common OS-specific ignores (e.g., `.DS_Store`, `Thumbs.db`).
        - Python ignores (e.g., `__pycache__/`, `*.pyc`, `*venv/`, `*env/`).
        - Node.js ignores (e.g., `node_modules/` globally).
        - Django-specific ignores (e.g., `db.sqlite3`, `media/` for user uploads, `staticfiles/` if collected).
        - Project-specific ignores identified in Tasks 2 & 3:
            - `sentrl_demo_webapp/ml_models/`
            - `sentrl_agentic_platform_tools/fine_tuned_models/` (if not committed directly).
            - Specific large data files/outputs from `sentrl_agentic_platform_tools` (e.g., `current-state-data_instruct.json`, `sample_actions.json` if anonymized versions are not committed, outputs from synthetic data or user logger).
        - Environment variables & secrets (e.g., `.env*`, `!.env.example`).
        - IDE & editor ignores (e.g., `.vscode/`, `.idea/`).
        - Sensitive directories like `certs/` if containing private keys.
    - **Action for User:**
        - Finalize the list of specific large data/model files/directories from `sentrl_agentic_platform_tools` for inclusion in `.gitignore`.
        - Decide on handling for `certs/` directories (commit templates vs. ignore private keys).
        - Decide on committing `package-lock.json`/`yarn.lock` (generally committed for apps).
- **4.2. Root Files:**
    - Ensure no PII, secrets, or unnecessary local files exist at the project root (`/Users/afrozmohammad/Documents/Sentrlai/Software/sentrlai/`).
    - **Action for User:** Review any files at the project root (other than the main project directories and `github_publish_review_tasks.md`) for suitability for public release.
- **4.3. Project Organization (Minor Refactoring if Needed):**
    - Address any significant redundancies or structural issues.
    - **Identified potential issues for user review & decision:**
        - **Redundant `certs/` and `media/` directories:** Present in `sentrl_demo_webapp/` and `sentrl_demo_webapp/sentrl_project/`. Consolidate to a single logical location or remove unused ones. Clarify `MEDIA_ROOT` setting.
        - **`chat_simulation_app/` in `sentrl_agentic_platform_tools/synthetic_data_generators/`**: Determine if this is distinct or a redundant copy of `sentrl_demo_webapp/chat_simulation_app/`. Plan for consolidation if redundant.
    - **Action for User:** Review overall directory structure for clarity and remove/refactor identified redundancies.

### Task 5: Licensing and Core Documentation
- **5.1. Add `LICENSE` File:**
    - Create a `LICENSE` file in the project root with the full GNU AGPLv3 license text.
    - **Action for User:** Obtain the official AGPLv3 license text and place it in `LICENSE` (or `LICENSE.md`/`LICENSE.txt`) at the project root.
- **5.2. Create Root `README.md`:** Draft a comprehensive README including:
    - **Project Title & Description:** Clearly state project's purpose (personal AI agents via task demonstration, Sentrl AI vision) and mention the AGPLv3 license.
    - **Features:** List key functionalities of `sentrl_demo_webapp` and `sentrl_agentic_platform_tools` (e.g., web UI, chat, user action logging, data processing, model tuning tools).
    - **Tech Stack:** Detail backend (Python, Django), frontend (JS, React), AI/ML libs (Transformers, PyTorch, Hugging Face, Ollama), databases (Qdrant, MongoDB), etc.
    - **Prerequisites:** List all necessary software (Python version, Node.js/npm version, pip, Git, other system dependencies).
    - **Installation/Setup Guide (Detailed Steps):**
        - Cloning the repository.
        - Python virtual environment setup.
        - **Environment Variables:** Instruct users to copy `.env.example` to `.env` and populate it with their specific secrets and configurations (Django secret key, DB credentials, API keys, etc.).
        - Backend (`sentrl_demo_webapp`) setup: `pip install -r requirements.txt`, `python manage.py migrate`, `python manage.py createsuperuser` (optional).
        - Frontend (`sentrl_demo_webapp`) setup: `npm install` (or `yarn`), `npm run build`.
        - `sentrl_agentic_platform_tools` setup: Instructions for installing dependencies (global requirements or per-tool, e.g., `pip install .` for packages like `synthetic_data_gen`).
        - Running development servers (Django, Node if applicable).
    - **Usage Guide:** How to run/use the webapp, an overview of `user_action_logger`, and brief on other agentic tools.
    - **Models:** If models are gitignored, provide clear instructions on how to download/train them and where to place them.
    - **Directory Structure Overview:** Brief explanation of main directories and their purpose.
    - **(Optional) Troubleshooting:** Common setup issues.
    - **Contributing:** Refer to `CONTRIBUTING.md`.
    - **License:** State AGPLv3 and link to `LICENSE` file.
    - **Action for User:** Gather all specific details needed for each README section (project name, feature list, exact prerequisites, setup commands, usage examples, model details, etc.).
- **5.3. Create `.env.example` File(s):**
    - Create a template file (`.env.example`) at the project root (or per-component if necessary).
    - List all required environment variables with placeholders/examples:
        - Django: `DJANGO_SECRET_KEY`, `DEBUG`, `ALLOWED_HOSTS`, Database configs (`DATABASE_ENGINE`, `DATABASE_NAME`, etc., or full `DATABASE_URL`).
        - API Keys: `OLLAMA_API_BASE_URL`, `OLLAMA_API_KEY`, keys for other external services (OpenAI, etc.).
        - Database connections for tools: `MONGODB_HOST`, `MONGODB_PORT`, `MONGODB_USERNAME`, `MONGODB_PASSWORD`, `MONGODB_DATABASE_NAME`, `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_API_KEY`.
        - Any other configurable parameters externalized from code.
    - **Action for User:** Compile a complete list of all environment variables used across the project and their placeholder values.
- **5.4. (Optional) Create `CONTRIBUTING.md`:**
    - Outline guidelines for bug reports, feature requests, coding standards, testing, and the Pull Request process.
    - Emphasize that PRs to the main repository are the preferred contribution method.
    - **Action for User:** Decide if this file is needed and draft its content.
- **5.5. (Optional) Create `CODE_OF_CONDUCT.md`:**
    - Add a standard Code of Conduct (e.g., Contributor Covenant).
    - **Action for User:** Decide if this file is needed and select/draft its content.

### Task 6: Data Handling and PII Protection for Users
- **6.1. Database:**
    - Confirm `db.sqlite3` (and any other database files, e.g., local Qdrant persistence if applicable) are not committed and are covered by `.gitignore`.
    - Ensure README (Task 5.2) includes clear instructions for database initialization (e.g., `python manage.py migrate`).
    - **Action for User:** Verbally reconfirm no live database files will be committed. Review if `sentrl_agentic_platform_tools` use any other local file-based databases that need gitignoring if they contain dynamic/PII data.
- **6.2. Sample Data/Screenshots for README:**
    - Plan illustrative, PII-free screenshots or mockups for the README to demonstrate application functionality without revealing sensitive data or detailed internal data pipelines.
    - **Screenshot/Mockup Ideas:**
        - `sentrl_demo_webapp`: Main UI, chatbot interaction (with generic examples), agent management interface (with placeholder data).
        - `user_action_logger`: Conceptual diagram or mocked UI showing types of actions logged generically.
        - Agent Building Flow: Conceptual flowchart illustrating the process from user action to agent personalization.
    - Include a statement in the README (e.g., under a "Data Privacy" or "Demo Data" section) clarifying that live databases/detailed pipelines are not shared and screenshots use mock/illustrative data.
    - **Action for User:** Decide which application parts to illustrate. Plan generation of PII-free screenshots/mockups. Note the data privacy statement for the README.

### Task 7: Architecture Documentation
- **7.1. Describe Combined Architecture in README (or separate `ARCHITECTURE.md`):**
    - Provide a clear, high-level textual description of how project components interact, focusing on the user-action-to-agent-personalization flow, aligning with the Sentrl AI vision.
    - **Key Components to Describe:**
        - **Frontend (`sentrl_demo_webapp`):** Web UI, Chatbot interface.
        - **Backend AI Agent (Conceptual Layer):** How user requests are processed, model interaction (fine-tuned models, Ollama, Qdrant for RAG).
        - **`user_action_logger`:** Its role in capturing user actions, what is captured (with PII caveats), and data flow (e.g., to MongoDB via loader).
        - **Data Processing & Storage Tools:** Role of `action_mongodb_loader`, `action_qdrant_loader`, data transformation, synthetic data generation, and (auto)labeling tools.
        - **Model Usage & Fine-tuning Tools:** How `fine_tuned_models/` are used, and how `action_instruction_tuning_formatting/` tools contribute to personalization.
        - **Core User Flow for Agent Personalization:** Detail the cycle from user action logging to agent learning/improvement and interaction.
    - **Format:**
        - Section in `README.md` or a separate `ARCHITECTURE.md` file.
        - **(Optional but Recommended) Diagram:** ASCII art or an embedded image (e.g., PNG from draw.io) showing components and data/request flows.
    - **Action for User:** Sketch out the architecture, decide on format (README vs. separate file), consider creating a diagram, and gather details on component interactions and the agent personalization workflow.

### Task 8: Git History Review (User Action)
- **8.1.** Advise user to thoroughly review Git commit history for any inadvertently committed PII or secrets.
    - **Guidance for User Review:**
        - Recall any past instances where PII/secrets might have been temporarily committed.
        - Use `git log -p -- <file_path>` to inspect changes in sensitive files (e.g., `settings.py`, configs, data files).
        - Use `git log -S "<string_to_search>"` to find commits introducing/removing specific known past secrets or PII patterns.
        - Consider using local secret scanning tools (e.g., `trufflehog`, `gitleaks`) on the repository before making it public.
- **8.2.** If sensitive data is found in history, recommend tools like BFG Repo-Cleaner or `git filter-branch` (with strong cautions) or suggest starting a fresh repository if history is too compromised. This is a manual step for the user.
    - **Remediation Options for User (CRITICAL: Back up repository before any history rewriting):**
        - **BFG Repo-Cleaner:** Generally recommended as safer/easier for removing unwanted data (secrets, large files). Rewrites history.
        - **`git filter-branch`:** Powerful but complex Git command for history rewriting. Use with extreme caution. Rewrites history.
        - **Start a Fresh Repository:** If history is very messy or rewriting is too risky. Involves deleting `.git` directory, `git init`, and making a fresh initial commit of the clean working directory (loses granular history).
    - **User Action:** Manually and carefully perform Git history review. If sensitive data is found, choose and execute a remediation strategy after backing up the repository.

## Phase 3: Final Review and Publishing

### Task 9: Final Checks
- **9.1.** Perform a final pass over all files to be committed.
    - **Action for User:** Use `git status` and `git diff` (or IDE tools) to review all staged files. Ensure all changes are intentional and clean.
- **9.2.** Ensure all PII and secrets are removed from committed files or properly externalized to an `.env` file (with `.env` itself being gitignored).
    - **Action for User:** Double-check key files identified in Tasks 2 & 3 (e.g., `settings.py`, configs, scripts, data files to be committed, README examples) for any residual PII/secrets.
- **9.3.** Validate that `.env.example` correctly reflects all required environment variables and that placeholder values are clear.
    - **Action for User:** Compare a working `.env` file (gitignored) against `.env.example` to ensure completeness and clarity of the template.
- **9.4.** Ensure all documentation (`README.md`, `LICENSE`, `ARCHITECTURE.md` (if created), `CONTRIBUTING.md` (if created)) is finalized, clear, and setup instructions are accurate and tested.
    - **Action for User:** Read through all documentation from a new user's perspective. If possible, test the setup/installation instructions in a clean environment.

### Task 10: User Actions for Publishing
- **10.1.** User to perform the Git history clean if necessary (as decided in Task 8).
    - **Context:** Ensures the commit history to be published is free of PII/secrets.
    - **Action for User:** If history rewriting was performed, ensure it's completed and the local repository is in the desired state. Always work on a fresh clone or from a backup when doing this.
- **10.2.** User to create a new public repository on GitHub (or chosen platform).
    - **Action for User:** Create a new, empty public repository on the hosting platform. Avoid initializing it with README, .gitignore, or license from the platform side if these are already prepared locally.
- **10.3.** User to push the cleaned and prepared project to the new public repository.
    - **Git Commands for User (execute in local project terminal):**
        - **If starting fresh locally (after `git init`):**
            - `git remote add origin <URL_of_new_GitHub_repo.git>`
            - `git branch -M main` (or `master`)
            - `git push -u origin main`
        - **If using existing local repo with cleaned history:**
            - `git remote set-url origin <URL_of_new_GitHub_repo.git>` (if `origin` remote exists and needs updating)
            - OR `git remote add origin <URL_of_new_GitHub_repo.git>` (if `origin` remote does not exist or adding as new)
            - `git push -u origin main` (use `--force` with extreme caution if repository history on remote differs and needs overwriting, e.g., after history rewrite or if remote was initialized with files. **Ensure pushing to the correct new public repo.**)
    - **Action for User:** Execute appropriate Git commands to push the local repository to the new public remote. Verify on the platform that all files and history are as expected.

This list will guide our process. 