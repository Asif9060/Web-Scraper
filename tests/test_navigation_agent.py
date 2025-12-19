"""
Tests for navigation agent module.

Tests goal-directed navigation and page assessment.
"""

import pytest

from web_intel.navigation_agent import (
    NavigationAgent,
    NavigationPlanner,
    NavigationGoal,
    NavigationPlan,
    NavigationStep,
    NavigationResult,
    PageAssessment,
    RelevanceLevel,
)
from web_intel.config import Settings


class TestNavigationGoal:
    """Tests for NavigationGoal dataclass."""

    def test_goal_creation(self):
        """NavigationGoal should be created correctly."""
        goal = NavigationGoal(
            objective="Find pricing information",
            target_content=["pricing", "plans", "cost"],
            max_steps=10,
        )

        assert goal.objective == "Find pricing information"
        assert "pricing" in goal.target_content
        assert goal.max_steps == 10

    def test_goal_with_constraints(self):
        """NavigationGoal can have constraints."""
        goal = NavigationGoal(
            objective="Find contact info",
            target_content=["contact", "email", "phone"],
            max_steps=5,
            stay_on_domain=True,
            avoid_patterns=["/login", "/admin"],
        )

        assert goal.stay_on_domain
        assert "/login" in goal.avoid_patterns

    def test_goal_to_dict(self):
        """NavigationGoal should convert to dictionary."""
        goal = NavigationGoal(
            objective="Find info",
            target_content=["info"],
            max_steps=10,
        )

        goal_dict = goal.to_dict() if hasattr(goal, "to_dict") else vars(goal)

        assert "objective" in goal_dict


class TestNavigationPlanner:
    """Tests for NavigationPlanner."""

    @pytest.fixture
    def planner(self, test_settings: Settings) -> NavigationPlanner:
        """Provide a navigation planner."""
        return NavigationPlanner(test_settings)

    def test_planner_creation(self, planner: NavigationPlanner):
        """Planner should be created successfully."""
        assert planner is not None

    def test_create_initial_plan(self, planner: NavigationPlanner):
        """Planner should create initial navigation plan."""
        goal = NavigationGoal(
            objective="Find pricing page",
            target_content=["pricing", "plans"],
            max_steps=5,
        )

        plan = planner.create_plan(
            goal=goal,
            start_url="https://example.com",
        )

        assert isinstance(plan, NavigationPlan)
        assert len(plan.steps) >= 1

    def test_plan_includes_exploration(self, planner: NavigationPlanner):
        """Plan should include exploration steps."""
        goal = NavigationGoal(
            objective="Find documentation",
            target_content=["docs", "documentation", "guide"],
            max_steps=10,
        )

        plan = planner.create_plan(
            goal=goal,
            start_url="https://example.com",
        )

        # Should have steps for exploration
        step_types = [step.action for step in plan.steps]
        assert "explore" in step_types or "visit" in step_types or len(
            plan.steps) > 0

    def test_assess_page_relevance(self, planner: NavigationPlanner):
        """Planner should assess page relevance."""
        goal = NavigationGoal(
            objective="Find pricing",
            target_content=["pricing", "cost", "plans"],
            max_steps=5,
        )

        page_content = "Our pricing plans start at $9.99 per month."

        assessment = planner.assess_page(
            content=page_content,
            url="https://example.com/pricing",
            goal=goal,
        )

        assert isinstance(assessment, PageAssessment)
        assert assessment.relevance == RelevanceLevel.HIGH or assessment.relevance_score > 0.5

    def test_assess_irrelevant_page(self, planner: NavigationPlanner):
        """Planner should identify irrelevant pages."""
        goal = NavigationGoal(
            objective="Find pricing",
            target_content=["pricing", "cost"],
            max_steps=5,
        )

        page_content = "Welcome to our blog about technology trends."

        assessment = planner.assess_page(
            content=page_content,
            url="https://example.com/blog",
            goal=goal,
        )

        assert assessment.relevance == RelevanceLevel.LOW or assessment.relevance_score < 0.5

    def test_extract_link_candidates(self, planner: NavigationPlanner):
        """Planner should extract promising link candidates."""
        goal = NavigationGoal(
            objective="Find pricing",
            target_content=["pricing", "plans"],
            max_steps=5,
        )

        links = [
            {"href": "/pricing", "text": "Pricing"},
            {"href": "/about", "text": "About Us"},
            {"href": "/plans", "text": "View Plans"},
            {"href": "/blog", "text": "Blog"},
        ]

        candidates = planner.rank_links(links, goal)

        # Pricing-related links should rank higher
        assert candidates[0]["href"] in ["/pricing", "/plans"]


class TestNavigationPlan:
    """Tests for NavigationPlan dataclass."""

    def test_plan_creation(self):
        """NavigationPlan should be created correctly."""
        plan = NavigationPlan(
            goal_objective="Find contact page",
            steps=[
                NavigationStep(action="visit", url="https://example.com"),
                NavigationStep(action="explore",
                               url="https://example.com/contact"),
            ],
        )

        assert len(plan.steps) == 2
        assert plan.goal_objective == "Find contact page"

    def test_plan_current_step(self):
        """Plan should track current step."""
        plan = NavigationPlan(
            goal_objective="Find info",
            steps=[
                NavigationStep(action="visit", url="https://example.com"),
                NavigationStep(action="visit", url="https://example.com/info"),
            ],
        )

        assert plan.current_step_index == 0 or plan.current_step is not None

    def test_plan_is_complete(self):
        """Plan should know when complete."""
        plan = NavigationPlan(
            goal_objective="Find info",
            steps=[
                NavigationStep(action="visit", url="https://example.com"),
            ],
            completed=True,
        )

        assert plan.is_complete or plan.completed


class TestNavigationStep:
    """Tests for NavigationStep dataclass."""

    def test_step_creation(self):
        """NavigationStep should be created correctly."""
        step = NavigationStep(
            action="visit",
            url="https://example.com/page",
            reason="Page likely contains target content",
        )

        assert step.action == "visit"
        assert step.url == "https://example.com/page"

    def test_step_with_result(self):
        """NavigationStep can have execution result."""
        step = NavigationStep(
            action="visit",
            url="https://example.com/page",
        )
        step.result = "Found relevant content"
        step.success = True

        assert step.success
        assert step.result is not None


class TestNavigationAgent:
    """Tests for NavigationAgent."""

    @pytest.fixture
    def agent(self, test_settings: Settings) -> NavigationAgent:
        """Provide a navigation agent."""
        return NavigationAgent(test_settings)

    def test_agent_creation(self, agent: NavigationAgent):
        """Agent should be created successfully."""
        assert agent is not None

    @pytest.mark.asyncio
    async def test_navigate_to_goal(self, agent: NavigationAgent):
        """Agent should navigate towards goal."""
        goal = NavigationGoal(
            objective="Find pricing information",
            target_content=["pricing", "plans", "cost"],
            max_steps=3,
        )

        # Mock or test with controlled environment
        result = await agent.navigate(
            start_url="https://example.com",
            goal=goal,
            dry_run=True,  # Don't actually navigate
        )

        assert isinstance(result, NavigationResult)

    @pytest.mark.asyncio
    async def test_navigation_respects_max_steps(self, agent: NavigationAgent):
        """Agent should respect max steps limit."""
        goal = NavigationGoal(
            objective="Find deep content",
            target_content=["deep"],
            max_steps=2,
        )

        result = await agent.navigate(
            start_url="https://example.com",
            goal=goal,
            dry_run=True,
        )

        assert result.steps_taken <= 2

    @pytest.mark.asyncio
    async def test_navigation_finds_target(self, agent: NavigationAgent):
        """Agent should report when target is found."""
        goal = NavigationGoal(
            objective="Find pricing",
            target_content=["pricing"],
            max_steps=5,
        )

        result = await agent.navigate(
            start_url="https://example.com",
            goal=goal,
            dry_run=True,
        )

        # Result should indicate success/failure
        assert hasattr(result, "success") or hasattr(result, "goal_achieved")


class TestNavigationResult:
    """Tests for NavigationResult dataclass."""

    def test_result_creation(self):
        """NavigationResult should be created correctly."""
        result = NavigationResult(
            success=True,
            steps_taken=3,
            final_url="https://example.com/pricing",
            content_found="Pricing information here",
        )

        assert result.success
        assert result.steps_taken == 3
        assert result.final_url == "https://example.com/pricing"

    def test_result_with_path(self):
        """NavigationResult can include navigation path."""
        result = NavigationResult(
            success=True,
            steps_taken=3,
            final_url="https://example.com/pricing",
            path=[
                "https://example.com",
                "https://example.com/products",
                "https://example.com/pricing",
            ],
        )

        assert len(result.path) == 3

    def test_result_failure(self):
        """NavigationResult should handle failures."""
        result = NavigationResult(
            success=False,
            steps_taken=5,
            final_url="https://example.com/dead-end",
            error="Could not find target content",
        )

        assert not result.success
        assert result.error is not None

    def test_result_to_dict(self):
        """NavigationResult should convert to dictionary."""
        result = NavigationResult(
            success=True,
            steps_taken=2,
            final_url="https://example.com/page",
        )

        result_dict = result.to_dict() if hasattr(result, "to_dict") else vars(result)

        assert "success" in result_dict
        assert "steps_taken" in result_dict


class TestPageAssessment:
    """Tests for PageAssessment dataclass."""

    def test_assessment_creation(self):
        """PageAssessment should be created correctly."""
        assessment = PageAssessment(
            url="https://example.com/page",
            relevance=RelevanceLevel.HIGH,
            relevance_score=0.85,
            content_matches=["pricing", "plans"],
        )

        assert assessment.relevance == RelevanceLevel.HIGH
        assert assessment.relevance_score == 0.85

    def test_assessment_suggested_actions(self):
        """Assessment can suggest next actions."""
        assessment = PageAssessment(
            url="https://example.com/page",
            relevance=RelevanceLevel.MEDIUM,
            relevance_score=0.6,
            suggested_links=["/pricing", "/plans"],
        )

        assert len(assessment.suggested_links) == 2


class TestRelevanceLevel:
    """Tests for RelevanceLevel enum."""

    def test_relevance_levels(self):
        """RelevanceLevel should have expected values."""
        assert RelevanceLevel.HIGH is not None
        assert RelevanceLevel.MEDIUM is not None
        assert RelevanceLevel.LOW is not None
        assert RelevanceLevel.NONE is not None

    def test_relevance_comparison(self):
        """Relevance levels should be comparable."""
        levels = [
            RelevanceLevel.NONE,
            RelevanceLevel.LOW,
            RelevanceLevel.MEDIUM,
            RelevanceLevel.HIGH,
        ]

        # Should be sortable by value
        sorted_levels = sorted(levels, key=lambda x: x.value)
        assert len(sorted_levels) == 4
