import json
from openai import OpenAI
from typing import List, Dict, Any


class LLMAsJudgeEvaluator:
    """Custom RAG evaluator using LLM-as-a-judge, matching frontend implementation"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model
    
    async def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Faithfulness: How consistent is the answer with the given context"""
        prompt = f"""Evaluiraj koliko je odgovor vjeran danom kontekstu. Ocijeni od 0 do 1 gdje:
0.0 = Potpuno proturječi kontekstu
0.25 = Uglavnom proturječi kontekstu  
0.5 = Neutralno/djelomično podržano
0.75 = Uglavnom podržano kontekstom
1.0 = Potpuno podržano kontekstom

Pitanje: {question}
Kontekst: {context}
Odgovor: {answer}

Odgovori u JSON formatu:
{{
  "score": <broj 0-1 s dvije decimale>,
  "reasoning": "<objašnjenje na hrvatskom>",
  "supported_claims": ["<tvrdnje podržane kontekstom>"],
  "unsupported_claims": ["<tvrdnje koje nisu podržane kontekstom>"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = content[content.find('{'):content.rfind('}')+1]
            result = json.loads(json_match)
            
            # Normalize score to 0-1 range
            if result["score"] > 1:
                result["score"] = (result["score"] - 1) / 4
            result["score"] = max(0.0, min(1.0, result["score"]))
            
            return result
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Greška u evaluaciji: {str(e)}",
                "supported_claims": [],
                "unsupported_claims": []
            }
    
    async def evaluate_answer_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        """Answer Relevancy: How relevant is the answer to the question"""
        prompt = f"""Evaluiraj koliko je odgovor relevantan za pitanje. Ocijeni od 0 do 1 gdje:
0.0 = Potpuno irelevantan
0.25 = Malo relevantan
0.5 = Umjereno relevantan  
0.75 = Vrlo relevantan
1.0 = Savršeno relevantan

Pitanje: {question}
Odgovor: {answer}

Odgovori u JSON formatu:
{{
  "score": <broj 0-1 s dvije decimale>,
  "reasoning": "<objašnjenje na hrvatskom>",
  "relevant_parts": ["<dijelovi odgovora koji su relevantni>"],
  "irrelevant_parts": ["<dijelovi odgovora koji su irelevantni>"]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content
            
            json_match = content[content.find('{'):content.rfind('}')+1]
            result = json.loads(json_match)
            
            if result["score"] > 1:
                result["score"] = (result["score"] - 1) / 4
            result["score"] = max(0.0, min(1.0, result["score"]))
            
            return result
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Greška u evaluaciji: {str(e)}",
                "relevant_parts": [],
                "irrelevant_parts": []
            }
    
    async def evaluate_context_precision(self, question: str, retrieved_docs: List[str]) -> Dict[str, Any]:
        """Context Precision: How precise is the retrieved context for the question"""
        context = "\n---\n".join(retrieved_docs)
        
        prompt = f"""Evaluiraj koliko je dohvaćeni kontekst precizan za odgovaranje na pitanje. Ocijeni od 0 do 1 gdje:
0.0 = Nema relevantnih informacija
0.25 = Malo relevantnih informacija
0.5 = Neke relevantne informacije
0.75 = Uglavnom relevantne informacije  
1.0 = Visoko relevantne i precizne informacije

Pitanje: {question}
Dohvaćeni kontekst: {context}

Odgovori u JSON formatu:
{{
  "score": <broj 0-1 s dvije decimale>,
  "reasoning": "<objašnjenje na hrvatskom>",
  "relevant_docs": [<indeksi relevantnih dokumenata>],
  "irrelevant_docs": [<indeksi irelevantnih dokumenata>],
  "missing_info": "<koje informacije nedostaju za potpun odgovor>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content
            
            json_match = content[content.find('{'):content.rfind('}')+1]
            result = json.loads(json_match)
            
            if result["score"] > 1:
                result["score"] = (result["score"] - 1) / 4
            result["score"] = max(0.0, min(1.0, result["score"]))
            
            return result
        except Exception as e:
            return {
                "score": 0.0,
                "reasoning": f"Greška u evaluaciji: {str(e)}",
                "relevant_docs": [],
                "irrelevant_docs": [],
                "missing_info": ""
            }