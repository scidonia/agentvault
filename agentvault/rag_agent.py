"""RAG Agent implementation using LangGraph."""

import json
from pathlib import Path
from typing import List, Dict, Any, TypedDict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from bookwyrm.client import BookWyrmClient

from .config import PROCESSED_DIR


class AgentState(TypedDict):
    """State for the RAG agent."""

    question: str
    search_results: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    answer: str
    messages: List[Any]


class RAGAgent:
    """RAG Agent using BookWyrm and LangGraph."""

    def __init__(self):
        self.bookwyrm = BookWyrmClient()
        self.knowledge_base = self._load_knowledge_base()
        self.graph = self._create_graph()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load processed texts from disk."""
        knowledge_base = []

        for json_file in PROCESSED_DIR.glob("*.json"):
            if json_file.name == "summary.json":
                continue

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                knowledge_base.extend(data["chunks"])

        print(f"Loaded {len(knowledge_base)} chunks from knowledge base")
        return knowledge_base

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("search", self.search_node)
        workflow.add_node("cite", self.citation_node)
        workflow.add_node("answer", self.answer_node)

        # Add edges
        workflow.set_entry_point("search")
        workflow.add_edge("search", "cite")
        workflow.add_edge("cite", "answer")
        workflow.add_edge("answer", END)

        return workflow.compile()

    def search_node(self, state: AgentState) -> AgentState:
        """Search for relevant chunks based on the question."""
        question = state["question"]
        print(f"Searching for: {question}")

        # Get question embedding
        question_embedding = self.bookwyrm.get_embedding(question)

        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(self.knowledge_base):
            chunk_embedding = np.array(chunk["embedding"])
            similarity = cosine_similarity([question_embedding], [chunk_embedding])[0][
                0
            ]

            similarities.append({"index": i, "similarity": similarity, "chunk": chunk})

        # Sort by similarity and take top 5
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:5]

        print(f"Found {len(top_results)} relevant chunks")
        for i, result in enumerate(top_results):
            chunk = result["chunk"]
            print(
                f"  {i+1}. {chunk.get('title', 'Unknown')} (similarity: {result['similarity']:.3f})"
            )

        state["search_results"] = top_results
        return state

    def citation_node(self, state: AgentState) -> AgentState:
        """Create citations from search results."""
        search_results = state["search_results"]
        citations = []

        for i, result in enumerate(search_results):
            chunk = result["chunk"]
            citation = {
                "id": i + 1,
                "title": chunk.get("title", "Unknown Title"),
                "author": chunk.get("author", "Unknown Author"),
                "text": (
                    chunk["text"][:200] + "..."
                    if len(chunk["text"]) > 200
                    else chunk["text"]
                ),
                "full_text": chunk["text"],
                "similarity": result["similarity"],
            }
            citations.append(citation)

        print(f"Created {len(citations)} citations")
        state["citations"] = citations
        return state

    def answer_node(self, state: AgentState) -> AgentState:
        """Generate answer based on citations."""
        question = state["question"]
        citations = state["citations"]

        # Combine relevant text from citations
        context_texts = []
        for citation in citations:
            context_texts.append(
                f"From '{citation['title']}' by {citation['author']}: {citation['full_text']}"
            )

        context = "\n\n".join(context_texts)

        # Generate answer using BookWyrm (if it has text generation capabilities)
        # For now, we'll create a simple extractive answer
        answer_parts = [
            f"Based on the available texts, here's what I found regarding '{question}':\n"
        ]

        for citation in citations:
            if citation["similarity"] > 0.3:  # Only include highly relevant citations
                answer_parts.append(
                    f"According to '{citation['title']}' by {citation['author']}: "
                    f"{citation['full_text'][:300]}{'...' if len(citation['full_text']) > 300 else ''}"
                )

        if len([c for c in citations if c["similarity"] > 0.3]) == 0:
            answer_parts.append(
                "I couldn't find highly relevant information in the available texts to answer your question."
            )

        answer = "\n\n".join(answer_parts)

        # Add citations
        answer += "\n\nSources:\n"
        for citation in citations:
            if citation["similarity"] > 0.2:
                answer += f"- {citation['title']} by {citation['author']} (relevance: {citation['similarity']:.2f})\n"

        state["answer"] = answer
        print("Generated answer with citations")
        return state

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG agent with a question."""
        initial_state = AgentState(
            question=question,
            search_results=[],
            citations=[],
            answer="",
            messages=[HumanMessage(content=question)],
        )

        # Run the graph
        result = self.graph.invoke(initial_state)

        return {
            "question": question,
            "answer": result["answer"],
            "citations": result["citations"],
            "search_results": len(result["search_results"]),
        }


def main():
    """Example usage of the RAG agent."""
    agent = RAGAgent()

    # Example queries
    questions = [
        "What is the main theme of the story?",
        "Who are the main characters?",
        "What happens at the end?",
        "What is the setting of the story?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print("=" * 60)

        result = agent.query(question)
        print(result["answer"])
        print(f"\nUsed {result['search_results']} search results")


if __name__ == "__main__":
    main()
