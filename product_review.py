import spacy
import pandas as pd
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import phonenumbers
from email_validator import validate_email, EmailNotValidError

class ResumeParser:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the resume parser with spaCy model

        Args:
            model_name: spaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model {model_name} not found. Please install it using:")
            print(f"python -m spacy download {model_name}")
            raise

        # Compile regex patterns for better performance
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
        self.linkedin_pattern = re.compile(r'linkedin\.com/in/[\w-]+', re.IGNORECASE)
        self.github_pattern = re.compile(r'github\.com/[\w-]+', re.IGNORECASE)

        # Common skill keywords (can be expanded)
        self.technical_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'express', 'node.js', 'bootstrap'],
            'database': ['mysql', 'postgresql', 'mongodb', 'sqlite', 'oracle', 'redis', 'elasticsearch'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'matplotlib', 'seaborn'],
            'tools': ['git', 'jenkins', 'jira', 'slack', 'visual studio', 'intellij', 'eclipse']
        }

        # Education keywords
        self.education_keywords = ['university', 'college', 'bachelor', 'master', 'phd', 'degree', 'diploma', 'certification']

        # Experience keywords
        self.experience_keywords = ['experience', 'work', 'employment', 'career', 'position', 'role', 'job']

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        contact_info = {}

        # Extract email
        emails = self.email_pattern.findall(text)
        contact_info['email'] = emails[0] if emails else None

        # Extract phone number
        phones = self.phone_pattern.findall(text)
        if phones:
            phone = ''.join(phones[0])
            contact_info['phone'] = phone
        else:
            contact_info['phone'] = None

        # Extract LinkedIn
        linkedin_matches = self.linkedin_pattern.findall(text)
        contact_info['linkedin'] = f"https://{linkedin_matches[0]}" if linkedin_matches else None

        # Extract GitHub
        github_matches = self.github_pattern.findall(text)
        contact_info['github'] = f"https://{github_matches[0]}" if github_matches else None

        return contact_info

    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name using NER"""
        doc = self.nlp(text[:500])  # Process first 500 chars for efficiency

        # Look for person entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()

        # Fallback: assume first line might contain name
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            # Simple heuristic: if first line has 2-4 words and no special chars, it's likely a name
            words = first_line.split()
            if 2 <= len(words) <= 4 and not any(char in first_line for char in '@.com()'):
                return first_line

        return None

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract technical skills from resume text"""
        text_lower = text.lower()
        found_skills = {}

        for category, skills in self.technical_skills.items():
            found_skills[category] = []
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills[category].append(skill)

        # Remove empty categories
        found_skills = {k: v for k, v in found_skills.items() if v}

        return found_skills

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information"""
        doc = self.nlp(text)
        education = []

        # Find sentences containing education keywords
        education_sentences = []
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in self.education_keywords):
                education_sentences.append(sent.text)

        # Extract organizations (universities, colleges)
        for sent in education_sentences:
            sent_doc = self.nlp(sent)
            institutions = []
            degrees = []

            for ent in sent_doc.ents:
                if ent.label_ == "ORG":
                    institutions.append(ent.text)

            # Look for degree patterns
            degree_patterns = [
                r'\b(bachelor|master|phd|b\.s\.|b\.a\.|m\.s\.|m\.a\.|ph\.d\.)\s+(?:of\s+)?(\w+(?:\s+\w+)*)',
                r'\b(bs|ba|ms|ma)\s+(?:in\s+)?(\w+(?:\s+\w+)*)',
                r'\bdegree\s+in\s+(\w+(?:\s+\w+)*)'
            ]

            for pattern in degree_patterns:
                matches = re.findall(pattern, sent, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        degrees.append(' '.join(match))
                    else:
                        degrees.append(match)

            if institutions or degrees:
                education.append({
                    'institution': institutions[0] if institutions else 'Unknown',
                    'degree': degrees[0] if degrees else 'Unknown',
                    'raw_text': sent
                })

        return education

    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience information"""
        doc = self.nlp(text)
        experiences = []

        # Find sentences with experience keywords and organizations
        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(keyword in sent_lower for keyword in self.experience_keywords):
                sent_doc = self.nlp(sent.text)

                # Extract organizations and job titles
                orgs = [ent.text for ent in sent_doc.ents if ent.label_ == "ORG"]

                # Look for job title patterns
                job_title_patterns = [
                    r'\b(manager|director|engineer|developer|analyst|coordinator|specialist|associate|senior|junior)\b',
                    r'\b(software|data|business|project|product|marketing|sales|hr)\s+(engineer|analyst|manager|developer)'
                ]

                titles = []
                for pattern in job_title_patterns:
                    matches = re.findall(pattern, sent.text, re.IGNORECASE)
                    titles.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])

                # Extract dates
                dates = [ent.text for ent in sent_doc.ents if ent.label_ == "DATE"]

                if orgs or titles:
                    experiences.append({
                        'company': orgs[0] if orgs else 'Unknown',
                        'position': titles[0] if titles else 'Unknown',
                        'dates': dates[0] if dates else 'Unknown',
                        'raw_text': sent.text
                    })

        return experiences

    def calculate_evaluation_metrics(self, parsed_data: Dict) -> Dict[str, float]:
        """Calculate evaluation metrics for the parsed resume"""
        metrics = {}

        # Completeness score (0-1)
        required_fields = ['name', 'email', 'phone', 'skills', 'education', 'experience']
        filled_fields = sum(1 for field in required_fields if parsed_data.get(field))
        metrics['completeness_score'] = filled_fields / len(required_fields)

        # Skills diversity score
        skill_categories = len(parsed_data.get('skills', {}))
        metrics['skills_diversity_score'] = min(skill_categories / 5, 1.0)  # Normalized to 5 categories

        # Experience count
        metrics['experience_count'] = len(parsed_data.get('experience', []))

        # Education count
        metrics['education_count'] = len(parsed_data.get('education', []))

        # Overall quality score
        metrics['quality_score'] = (
            metrics['completeness_score'] * 0.4 +
            metrics['skills_diversity_score'] * 0.3 +
            min(metrics['experience_count'] / 3, 1.0) * 0.2 +
            min(metrics['education_count'] / 2, 1.0) * 0.1
        )

        return metrics

    def parse_resume(self, resume_text: str) -> Dict:
        """Main method to parse resume and extract all information"""
        parsed_data = {}

        # Extract basic information
        parsed_data['name'] = self.extract_name(resume_text)
        parsed_data.update(self.extract_contact_info(resume_text))

        # Extract structured information
        parsed_data['skills'] = self.extract_skills(resume_text)
        parsed_data['education'] = self.extract_education(resume_text)
        parsed_data['experience'] = self.extract_experience(resume_text)

        # Calculate metrics
        parsed_data['evaluation_metrics'] = self.calculate_evaluation_metrics(parsed_data)

        # Add metadata
        parsed_data['parsed_at'] = datetime.now().isoformat()
        parsed_data['parser_version'] = '1.0'

        return parsed_data

    def to_dataframe(self, parsed_resumes: List[Dict]) -> pd.DataFrame:
        """Convert parsed resume data to pandas DataFrame"""
        flattened_data = []

        for resume in parsed_resumes:
            flat_resume = {
                'name': resume.get('name'),
                'email': resume.get('email'),
                'phone': resume.get('phone'),
                'linkedin': resume.get('linkedin'),
                'github': resume.get('github'),
                'skills_count': sum(len(skills) for skills in resume.get('skills', {}).values()),
                'education_count': len(resume.get('education', [])),
                'experience_count': len(resume.get('experience', [])),
                'completeness_score': resume.get('evaluation_metrics', {}).get('completeness_score', 0),
                'quality_score': resume.get('evaluation_metrics', {}).get('quality_score', 0),
                'parsed_at': resume.get('parsed_at')
            }
            flattened_data.append(flat_resume)

        return pd.DataFrame(flattened_data)

    def export_results(self, parsed_data: Dict, filename: str, format: str = 'json'):
        """Export parsed results to file"""
        if format == 'json':
            with open(f"{filename}.json", 'w') as f:
                json.dump(parsed_data, f, indent=2)
        elif format == 'csv' and isinstance(parsed_data, list):
            df = self.to_dataframe(parsed_data)
            df.to_csv(f"{filename}.csv", index=False)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'")

# Example usage
def main():
    # Sample resume text for testing
    sample_resume = """
    John Doe
    Email: john.doe@email.com
    Phone: (555) 123-4567
    LinkedIn: linkedin.com/in/johndoe
    GitHub: github.com/johndoe

    EXPERIENCE
    Senior Software Engineer at Google Inc.
    2020 - Present
    - Developed scalable web applications using Python and Django
    - Led a team of 5 developers in building microservices architecture

    Software Developer at Microsoft Corp.
    2018 - 2020
    - Built REST APIs using Node.js and Express
    - Worked with React for frontend development

    EDUCATION
    Master of Science in Computer Science
    Stanford University, 2018

    Bachelor of Science in Software Engineering
    University of California, 2016

    SKILLS
    Programming Languages: Python, JavaScript, Java, C++
    Web Technologies: React, Angular, Django, Flask
    Databases: MySQL, PostgreSQL, MongoDB
    Cloud Platforms: AWS, Azure
    """

    # Initialize parser
    parser = ResumeParser()

    try:
        # Parse the resume
        result = parser.parse_resume(sample_resume)

        # Print results
        print("Parsed Resume Data:")
        print(json.dumps(result, indent=2))

        # Convert to DataFrame
        df = parser.to_dataframe([result])
        print("\nDataFrame Summary:")
        print(df)

        # Export results
        parser.export_results(result, "parsed_resume", "json")
        print("\nResults exported to parsed_resume.json")

    except Exception as e:
        print(f"Error parsing resume: {e}")

if __name__ == "__main__":
    main()