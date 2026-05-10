from crewai import Crew, Process
from agents import get_agents
from tasks import get_tasks
from dotenv import load_dotenv

load_dotenv()

class KingsGuardCrew:
    def __init__(self, prompt: str, agent_id: str = "default_user"):
        self.prompt = prompt
        self.agent_id = agent_id

    def create_crew(self) -> Crew:
        agents = list(get_agents())
        tasks = get_tasks(self.prompt, self.agent_id)

        # Assemble the crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            memory=True,  # Enables short-term memory (Security Context) and long-term memory
            verbose=True
        )
        return crew

    def kickoff(self):
        crew = self.create_crew()
        result = crew.kickoff()
        return result
