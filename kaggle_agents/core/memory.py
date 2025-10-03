"""Memory management for agent experiences and feedback."""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class Memory:
    """Manage agent memory and experience storage."""

    def __init__(self, competition_dir: str):
        """Initialize memory system.

        Args:
            competition_dir: Directory for competition-specific memory
        """
        self.competition_dir = Path(competition_dir)
        self.memory_file = self.competition_dir / "memory.json"
        self.experiences: List[Dict[str, Any]] = []

        # Load existing memory if available
        self.load()

    def add_experience(
        self,
        phase: str,
        agent_role: str,
        experience: Dict[str, Any]
    ):
        """Add an agent experience to memory.

        Args:
            phase: Workflow phase name
            agent_role: Role of the agent
            experience: Experience data including results, feedback, etc.
        """
        experience_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "agent_role": agent_role,
            "experience": experience
        }

        self.experiences.append(experience_entry)
        logger.info(f"Added experience for {agent_role} in {phase}")

    def get_experiences_by_agent(
        self,
        agent_role: str,
        phase: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all experiences for a specific agent.

        Args:
            agent_role: Role of the agent
            phase: Optional phase filter

        Returns:
            List of experience entries
        """
        experiences = [
            exp for exp in self.experiences
            if exp['agent_role'] == agent_role
        ]

        if phase:
            experiences = [
                exp for exp in experiences
                if exp['phase'] == phase
            ]

        return experiences

    def get_experiences_by_phase(self, phase: str) -> List[Dict[str, Any]]:
        """Get all experiences for a specific phase.

        Args:
            phase: Workflow phase name

        Returns:
            List of experience entries
        """
        return [
            exp for exp in self.experiences
            if exp['phase'] == phase
        ]

    def get_last_experience(
        self,
        agent_role: str,
        phase: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent experience for an agent.

        Args:
            agent_role: Role of the agent
            phase: Optional phase filter

        Returns:
            Most recent experience or None
        """
        experiences = self.get_experiences_by_agent(agent_role, phase)

        if experiences:
            return experiences[-1]
        return None

    def get_feedback_for_agent(
        self,
        agent_role: str,
        phase: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all reviewer feedback for an agent.

        Args:
            agent_role: Role of the agent
            phase: Optional phase filter

        Returns:
            List of feedback entries
        """
        experiences = self.get_experiences_by_agent(agent_role, phase)
        feedback = []

        for exp in experiences:
            if 'reviewer_feedback' in exp['experience']:
                feedback.append({
                    'timestamp': exp['timestamp'],
                    'phase': exp['phase'],
                    'feedback': exp['experience']['reviewer_feedback']
                })

        return feedback

    def get_successful_experiences(
        self,
        agent_role: str,
        min_score: float = 3.0
    ) -> List[Dict[str, Any]]:
        """Get experiences that received good scores.

        Args:
            agent_role: Role of the agent
            min_score: Minimum score threshold

        Returns:
            List of successful experiences
        """
        experiences = self.get_experiences_by_agent(agent_role)
        successful = []

        for exp in experiences:
            feedback = exp['experience'].get('reviewer_feedback', {})
            score = feedback.get('score', 0)

            if score >= min_score:
                successful.append(exp)

        return successful

    def save(self):
        """Save memory to disk."""
        try:
            self.competition_dir.mkdir(parents=True, exist_ok=True)

            with open(self.memory_file, 'w') as f:
                json.dump(self.experiences, f, indent=2)

            logger.info(f"Memory saved to {self.memory_file}")

        except Exception as e:
            logger.error(f"Error saving memory: {e}")

    def load(self):
        """Load memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    self.experiences = json.load(f)

                logger.info(f"Loaded {len(self.experiences)} experiences from memory")

            except Exception as e:
                logger.error(f"Error loading memory: {e}")
                self.experiences = []
        else:
            logger.info("No existing memory file found, starting fresh")
            self.experiences = []

    def clear(self):
        """Clear all memory."""
        self.experiences = []
        if self.memory_file.exists():
            self.memory_file.unlink()
        logger.info("Memory cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored experiences.

        Returns:
            Dictionary with memory statistics
        """
        if not self.experiences:
            return {
                "total_experiences": 0,
                "agents": [],
                "phases": []
            }

        agents = list(set(exp['agent_role'] for exp in self.experiences))
        phases = list(set(exp['phase'] for exp in self.experiences))

        # Calculate average scores by agent
        agent_scores = {}
        for agent in agents:
            scores = []
            for exp in self.experiences:
                if exp['agent_role'] == agent:
                    feedback = exp['experience'].get('reviewer_feedback', {})
                    score = feedback.get('score')
                    if score is not None:
                        scores.append(score)

            if scores:
                agent_scores[agent] = sum(scores) / len(scores)

        return {
            "total_experiences": len(self.experiences),
            "agents": agents,
            "phases": phases,
            "agent_average_scores": agent_scores,
            "experiences_per_agent": {
                agent: len(self.get_experiences_by_agent(agent))
                for agent in agents
            },
            "experiences_per_phase": {
                phase: len(self.get_experiences_by_phase(phase))
                for phase in phases
            }
        }

    def export_summary(self, output_file: Optional[Path] = None) -> str:
        """Export a human-readable summary of memory.

        Args:
            output_file: Optional file to write summary to

        Returns:
            Summary string
        """
        stats = self.get_statistics()

        summary = f"""# Memory Summary

## Overview
- Total Experiences: {stats['total_experiences']}
- Agents: {', '.join(stats['agents'])}
- Phases: {', '.join(stats['phases'])}

## Agent Performance
"""

        for agent, avg_score in stats.get('agent_average_scores', {}).items():
            count = stats['experiences_per_agent'][agent]
            summary += f"- {agent}: {avg_score:.2f} average score ({count} experiences)\n"

        summary += "\n## Phase Coverage\n"
        for phase, count in stats['experiences_per_phase'].items():
            summary += f"- {phase}: {count} experiences\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(summary)
            logger.info(f"Summary exported to {output_file}")

        return summary


if __name__ == '__main__':
    # Test memory system
    memory = Memory("test_competition")

    # Add some test experiences
    memory.add_experience(
        phase="Data Cleaning",
        agent_role="planner",
        experience={
            "result": "Created plan with 5 steps",
            "reviewer_feedback": {
                "score": 4.5,
                "suggestion": "Consider handling outliers"
            }
        }
    )

    memory.add_experience(
        phase="Data Cleaning",
        agent_role="developer",
        experience={
            "result": "Implemented data cleaning code",
            "reviewer_feedback": {
                "score": 4.0,
                "suggestion": "Add more error handling"
            }
        }
    )

    # Save and print statistics
    memory.save()
    print("\n" + memory.export_summary())

    # Test retrieval
    planner_experiences = memory.get_experiences_by_agent("planner")
    print(f"\nPlanner has {len(planner_experiences)} experiences")
