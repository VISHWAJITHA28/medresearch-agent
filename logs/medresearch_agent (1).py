"""
MedResearch Agent - Medical Research Paper Analyzer

This agent processes medical research papers, generates plain-language summaries,
performs multi-paper synthesis, and provides evidence-based Q&A.

Author: Your Name <your.email@example.com>
License: Apache 2.0
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import sys
import os

from bindu.penguin.bindufy import bindufy

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

# Import custom tools - Try both ways
try:
    from tools.pdf_processor import PDFProcessor
    from tools.medical_nlp import MedicalNLP
    from tools.evidence_scorer import EvidenceScorer
    from tools.synthesizer import PaperSynthesizer
    logger = logging.getLogger(__name__)
    logger.info("✅ Imported from tools/ directory")
except ImportError:
    try:
        from pdf_processor import PDFProcessor
        from medical_nlp import MedicalNLP
        from evidence_scorer import EvidenceScorer
        from synthesizer import PaperSynthesizer
        logger = logging.getLogger(__name__)
        logger.info("✅ Imported from current directory")
    except ImportError as e:
        print(f"❌ ERROR: Could not import tools: {e}")
        print("Make sure pdf_processor.py, medical_nlp.py, evidence_scorer.py, and synthesizer.py exist!")
        sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Medical disclaimer to include in all responses
MEDICAL_DISCLAIMER = """
⚠️ **MEDICAL DISCLAIMER**
This information is for educational and research purposes only.
It is NOT medical advice and should NOT be used for diagnosis or treatment.
Always consult qualified healthcare professionals for medical decisions.
"""


class MedResearchAgent:
    """Medical Research Analysis Agent with multi-paper synthesis capabilities."""

    def __init__(self):
        """Initialize the agent with all necessary tools."""
        self.pdf_processor = PDFProcessor()
        self.medical_nlp = MedicalNLP()
        self.evidence_scorer = EvidenceScorer()
        self.synthesizer = PaperSynthesizer()
        self.paper_cache: Dict[str, Dict[str, Any]] = {}

        logger.info("MedResearch Agent initialized successfully")

    def process_uploaded_papers(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Extract and process any PDF papers from uploaded files.

        Args:
            messages: List of message dictionaries

        Returns:
            List of processed paper data dictionaries
        """
        papers = []

        # Check for uploaded files (this is a placeholder - actual implementation
        # would need to extract files from the Bindu message format)
        for message in messages:
            if message.get("role") == "user" and "attachments" in message:
                for attachment in message["attachments"]:
                    if attachment.get("type") == "application/pdf":
                        try:
                            paper_data = self.pdf_processor.process_paper(
                                attachment["path"]
                            )
                            papers.append(paper_data)
                            self.paper_cache[paper_data["id"]] = paper_data
                            logger.info(f"Processed paper: {paper_data['title']}")
                        except Exception as e:
                            logger.error(f"Error processing PDF: {e}")

        return papers

    def generate_summary(self, paper_data: Dict[str, Any]) -> str:
        """
        Generate a plain-language summary of a research paper.

        Args:
            paper_data: Extracted paper information

        Returns:
            Plain-language summary with key findings
        """
        # Extract medical entities
        entities = self.medical_nlp.extract_entities(paper_data["full_text"])

        # Score evidence quality
        quality_score = self.evidence_scorer.score_paper(paper_data)

        summary = f"""
📄 **{paper_data['title']}**

👥 **Authors:** {', '.join(paper_data['authors'])}
📅 **Published:** {paper_data['year']}
⭐ **Evidence Quality Score:** {quality_score}/100

**Plain-Language Summary:**
{paper_data['abstract_summary']}

**Key Findings:**
{self._format_findings(paper_data['key_findings'])}

**Study Design:**
- Type: {paper_data['study_type']}
- Sample Size: {paper_data['sample_size']}
- Duration: {paper_data.get('duration', 'Not specified')}

**Medical Entities Detected:**
- Conditions: {', '.join(entities.get('conditions', [])[:5])}
- Medications: {', '.join(entities.get('medications', [])[:5])}
- Procedures: {', '.join(entities.get('procedures', [])[:5])}

**Limitations:**
{self._format_limitations(paper_data.get('limitations', []))}

{MEDICAL_DISCLAIMER}
"""
        return summary.strip()

    def synthesize_multiple_papers(
        self, paper_ids: List[str], topic: str
    ) -> str:
        """
        Compare and synthesize findings from multiple papers.

        Args:
            paper_ids: List of paper IDs to synthesize
            topic: Research topic for context

        Returns:
            Multi-paper synthesis report
        """
        papers = [self.paper_cache[pid] for pid in paper_ids if pid in self.paper_cache]

        if len(papers) < 2:
            return "⚠️ Need at least 2 papers for synthesis. Please upload more papers."

        synthesis = self.synthesizer.synthesize(papers, topic)

        report = f"""
🔬 **Multi-Paper Synthesis: {topic}**

📚 **Papers Analyzed:** {len(papers)}

**Consensus Findings:**
{self._format_findings(synthesis['consensus_findings'])}

**Areas of Agreement:**
{self._format_list(synthesis['agreements'])}

**Contradictions & Debates:**
{self._format_contradictions(synthesis['contradictions'])}

**Overall Evidence Strength:** {synthesis['overall_evidence_strength']}/100

**Confidence Level:** {synthesis['confidence_level']}

**Recommendations for Further Research:**
{self._format_list(synthesis['research_gaps'])}

{MEDICAL_DISCLAIMER}
"""
        return report.strip()

    def answer_question(self, question: str, context_papers: List[str]) -> str:
        """
        Answer a question based on uploaded papers with citations.

        Args:
            question: User's question
            context_papers: List of paper IDs to search

        Returns:
            Answer with citations
        """
        relevant_papers = [
            self.paper_cache[pid] for pid in context_papers if pid in self.paper_cache
        ]

        if not relevant_papers:
            return "⚠️ No papers available. Please upload research papers first."

        # Find relevant sections (simple implementation - can be enhanced)
        answer_parts = []
        citations = []

        for paper in relevant_papers:
            # Simple keyword matching (can be enhanced with semantic search)
            if any(
                keyword.lower() in paper["full_text"].lower()
                for keyword in question.split()[:5]
            ):
                answer_parts.append(
                    f"According to {paper['authors'][0]} et al. ({paper['year']}): "
                    f"{paper['key_findings'][0] if paper['key_findings'] else 'Relevant information found.'}"
                )
                citations.append(
                    f"[{len(citations) + 1}] {paper['title']} - {paper['authors'][0]} et al., {paper['year']}"
                )

        if not answer_parts:
            return f"⚠️ No relevant information found in the uploaded papers for: '{question}'"

        answer = f"""
**Question:** {question}

**Answer:**
{' '.join(answer_parts)}

**Citations:**
{chr(10).join(citations)}

{MEDICAL_DISCLAIMER}
"""
        return answer.strip()

    def _format_findings(self, findings: List[str]) -> str:
        """Format findings as numbered list."""
        return "\n".join(f"{i+1}. {finding}" for i, finding in enumerate(findings))

    def _format_limitations(self, limitations: List[str]) -> str:
        """Format limitations as bullet points."""
        if not limitations:
            return "- None specified"
        return "\n".join(f"- {lim}" for lim in limitations)

    def _format_list(self, items: List[str]) -> str:
        """Format list items as bullet points."""
        return "\n".join(f"• {item}" for item in items)

    def _format_contradictions(self, contradictions: List[Dict[str, str]]) -> str:
        """Format contradictions with papers."""
        if not contradictions:
            return "• No major contradictions found across studies"

        formatted = []
        for contra in contradictions:
            formatted.append(
                f"• **Issue:** {contra['issue']}\n"
                f"  - Paper A: {contra['position_a']}\n"
                f"  - Paper B: {contra['position_b']}"
            )
        return "\n\n".join(formatted)


# Initialize agent instance
med_agent = MedResearchAgent()


def handler(messages: List[Dict[str, str]]) -> str:
    """
    Main handler function for Bindu integration.
    FIXED VERSION - Responds to different questions!
    """
    try:
        # Get the latest user message
        user_message = messages[-1]["content"]
        user_message_lower = user_message.lower()
        
        # LOG what we received
        logger.info(f"=== RECEIVED: '{user_message}' ===")

        # Process any uploaded papers FIRST
        uploaded_papers = med_agent.process_uploaded_papers(messages)

        if uploaded_papers:
            logger.info("Processing uploaded papers")
            if len(uploaded_papers) == 1:
                return med_agent.generate_summary(uploaded_papers[0])
            else:
                paper_ids = [p["id"] for p in uploaded_papers]
                return med_agent.synthesize_multiple_papers(paper_ids, "uploaded research")

        # 1. GREETINGS
        if any(word in user_message_lower for word in ["hello", "hi", "hey", "greetings"]):
            logger.info("✅ Greeting detected")
            return """👋 **Hello! Welcome to MedResearch Agent!**

I'm an AI specialized in analyzing medical research papers.

**What I can do:**
📄 Analyze single papers with quality scores
🔬 Compare multiple papers and find consensus
❓ Answer questions with citations

**Try asking:**
• "What features do you have?"
• "How do you work?"
• "Tell me about yourself"

Or upload research papers to begin!

⚠️ MEDICAL DISCLAIMER: For educational purposes only. Not medical advice. Consult healthcare professionals for medical decisions."""

        # 2. FEATURES
        elif any(word in user_message_lower for word in ["feature", "capability", "what can you", "what do you"]):
            logger.info("✅ Features question detected")
            return """🌟 **My Capabilities:**

**1. PDF Paper Analysis** 📄
   • Extract key findings
   • Score evidence quality (0-100)
   • Identify medical entities
   • Detect study type and biases

**2. Multi-Paper Synthesis** 🔬
   • Compare findings across studies
   • Find consensus and contradictions
   • Generate meta-analysis
   • Calculate overall evidence strength

**3. Medical NLP** 🧬
   • Extract conditions (diabetes, hypertension)
   • Identify medications (metformin, aspirin)
   • Recognize procedures (surgery, MRI)
   • Find lab values and statistics

**4. Q&A with Citations** ❓
   • Answer questions about papers
   • Provide evidence-based responses
   • Include proper citations

Upload papers to get started!

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 3. HOW IT WORKS
        elif "how" in user_message_lower and "work" in user_message_lower:
            logger.info("✅ How it works detected")
            return """🔍 **How I Work:**

**Step 1: Upload** 📤
You upload medical research papers (PDF format)

**Step 2: Extract** 📑
I extract text, metadata, and study details

**Step 3: Analyze** 🔬
I perform:
• Medical entity recognition
• Key findings identification
• Evidence quality scoring
• Bias detection

**Step 4: Synthesize** 💡
For multiple papers:
• Compare methodologies
• Find consensus
• Identify contradictions
• Calculate overall evidence

**Step 5: Respond** 💬
I provide:
• Plain-language summaries
• Evidence-based answers
• Proper citations

**Technology:** NLP + Medical databases + Evidence-Based Medicine

Upload a paper to see it in action!

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 4. ABOUT
        elif any(phrase in user_message_lower for phrase in ["who are you", "what are you", "about you"]):
            logger.info("✅ About question detected")
            return """🏥 **About Me:**

I'm **MedResearch Agent** - an AI specialized in medical research paper analysis.

**My Mission:**
Help researchers, clinicians, and students navigate medical literature efficiently.

**What Makes Me Special:**
✨ Multi-paper synthesis (compare findings)
✨ Evidence quality scoring (objective ratings)
✨ Medical NLP (extract medical entities)
✨ Citation-backed answers
✨ Bias detection

**Built With:** Bindu + Python + NLP + Evidence-Based Medicine

**Safety:** I always include medical disclaimers and never provide direct medical advice.

Ready to analyze papers?

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 5. COMPARE/SYNTHESIS
        elif "compare" in user_message_lower or "synthesize" in user_message_lower:
            logger.info("✅ Synthesis request detected")
            paper_ids = list(med_agent.paper_cache.keys())
            if len(paper_ids) >= 2:
                return med_agent.synthesize_multiple_papers(paper_ids, user_message)
            else:
                return f"""📚 **Multi-Paper Synthesis**

Need at least 2 papers to compare. You have {len(paper_ids)} paper(s).

**How to use:**
1. Upload 2+ research papers (PDF)
2. Ask "compare papers" or "synthesize"
3. I'll show consensus and contradictions

Upload papers to get started!

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 6. QUESTIONS
        elif "?" in user_message or any(word in user_message_lower for word in ["what", "why", "when", "where"]):
            logger.info("✅ Question detected")
            paper_ids = list(med_agent.paper_cache.keys())
            if paper_ids:
                return med_agent.answer_question(user_message, paper_ids)
            else:
                return """❓ **I'd love to answer!**

But I need papers first:
1. Upload PDFs in "Upload Papers" tab
2. Wait for processing
3. Come back and ask again!

**Or ask about me:**
• "What can you do?"
• "How do you work?"

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 7. HELP
        elif any(word in user_message_lower for word in ["help", "guide", "instructions"]):
            logger.info("✅ Help request detected")
            return """📖 **How to Use:**

**1. Upload Papers** 📤
   Go to "Upload Papers" tab → Drag PDFs

**2. View Analysis** 📊
   Check "Analysis" tab for summaries

**3. Compare Papers** 🔬
   Upload 2+ papers → "Synthesis" tab

**4. Ask Questions** 💬
   Come here and ask anything!

**Try asking:**
• "What can you do?"
• "How do you work?"

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

        # 8. DEFAULT
        else:
            logger.info("⚠️ Default response")
            return """👋 **Welcome to MedResearch Agent!**

**Try asking:**
• "hi" or "hello"
• "what can you do?"
• "how do you work?"
• "help"

Or upload research papers to analyze!

⚠️ MEDICAL DISCLAIMER: For educational purposes only."""

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return f"⚠️ Error: {str(e)}"


# Configuration for Bindu
config = {
    "author": "your.email@example.com",
    "name": "medresearch_agent",
    "description": "Medical research paper analyzer with multi-paper synthesis and evidence scoring",
    "version": "1.0.0",
    "capabilities": {
        "pdf_processing": True,
        "multi_paper_synthesis": True,
        "evidence_scoring": True,
        "medical_nlp": True,
    },
    "auth": {"enabled": False},
    "storage": {"type": "memory"},
    "scheduler": {"type": "memory"},
    "deployment": {"url": "http://localhost:3773", "expose": True},
}

if __name__ == "__main__":
    logger.info("Starting MedResearch Agent...")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path[:3]}")
    bindufy(config, handler)
    logger.info("MedResearch Agent is live at http://localhost:3773")
