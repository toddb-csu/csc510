# Todd Bartoszkiewicz
# CSC510: Foundations of Artificial Intelligence
# Module 8: Porfolio Project
#
# Your final Portfolio Project will be a fully-functioning AI program built to solve a real-world problem of your
# choosing, utilizing the tools and techniques outlined in this course. Your program will interact with human beings to
# support decision-making processes by delivering relevant information about the problem.
#
# Your final project submission should include a self-executable Python program. The program should be complete and
# straightforward to test. The program should leverage methods learned from at least 2 of the modules from this course.
# The submission must function and be a reasonable attempt at a solution for your chosen problem. The solution does not
# have to be correct or useful in the real world, but the solution MUST provide reasonable answers without error.
#
# In addition to your program, your submission should include a 2-4 page essay describing the final version of your AI
# program, the use-case it intends to solve, and the methods you used toward that goal. In your paper, please address
# the following details:
#
# The tools, libraries, and APIs utilized
# Search methods used and how they contributed toward the program goal
# Inclusion of any deep learning models
# Aspects of your program that utilize expert system concepts
# How your program represent knowledge
# How symbolic planning is used in your program (remember, symbolic planning is not limited to robot navigation!)
# Important : Your final submission will also be required to include at least 3 references, from which your work is
# based. The CSU Global Library is a good place to find references. The CSU Global Writing Center offers resources on
# how to format your assignment and cite sources in APA style. The CSU Global Library and Writing Center links can be
# found in the course navigation panel.
#
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import networkx as nx
import sys

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

career_graph = nx.DiGraph()
career_graph.add_node("Junior Developer", skills=["Python", "SQL"])
career_graph.add_node("Senior Developer", skills=["Python", "SQL", "AWS", "Leadership"])
career_graph.add_node("Tech Lead", skills=["Python", "SQL", "AWS", "Leadership", "Project Management"])
career_graph.add_node("Engineering Manager", skills=["Leadership", "Project Management", "Budgeting"])
career_graph.add_node("Data Analyst", skills=["SQL", "Excel", "Data Visualization"])
career_graph.add_node("Data Scientist", skills=["SQL", "Python", "Machine Learning"])
career_graph.add_node("Product Manager", skills=["Project Management", "Market Analysis"])

career_graph.add_edge("Junior Developer", "Senior Developer", required_skills=["AWS", "Leadership"])
career_graph.add_edge("Senior Developer", "Tech Lead", required_skills=["Project Management"])
career_graph.add_edge("Tech Lead", "Engineering Manager", required_skills=["Project Management", "Resource Planning"])
career_graph.add_edge("Junior Developer", "Data Analyst", required_skills=["Excel", "Data Visualization"])
career_graph.add_edge("Data Analyst", "Data Scientist", required_skills=["Python", "Machine Learning"])
career_graph.add_edge("Senior Developer", "Product Manager", required_skills=["Project Management", "SWOT Analysis"])

open_positions = {
    "Open Senior Developer Position": {"required_skills": ["Python", "SQL", "AWS", "Leadership", "Agile Methodology"],
                                       "related_to": "Senior Developer"},
    "Open Data Analyst Position": {"required_skills": ["SQL", "Excel", "Data Visualization", "Agile Methodology"],
                                   "related_to": "Data Analyst"},
    "Open Tech Lead Position": {"required_skills": ["Python", "SQL", "AWS", "Leadership", "Agile Methodology"],
                                "related_to": "Tech Lead"},
    "Open Data Scientist Position": {"required_skills": ["SQL", "Python", "Machine Learning", "Data Visualization", "Agile Methodology"],
                                     "related_to": "Data Scientist"},
    "Open Product Manager Position": {"required_skills": ["Project Management", "SWOT Analysis", "Leadership", "Agile Methodology"],
                                      "related_to": "Product Manager"},
    "Open Engineering Manager Position": {"required_skills": ["Leadership", "Project Management", "Resource Planning"],
                                          "related_to": "Engineering Manager"}
}


def identify_skill_gaps(current_skills, required_skills):
    skill_gaps = [skill for skill in required_skills if skill not in current_skills]
    if not skill_gaps:
        return "No skill gaps identified."
    return f"Skill gaps: {', '.join(skill_gaps)}. Consider training in these areas before pursuing this role."


def compute_match_percentage(current_skills, required_skills):
    matching = len([skill for skill in required_skills if skill in current_skills])
    total = len(required_skills)
    return (matching / total * 100) if total > 0 else 0


# Traverse career graph to map path from current role to desired role.
def plan_career_path(current_role, target_interest):
    if current_role not in career_graph.nodes:
        return None
    try:
        path = nx.shortest_path(career_graph, current_role, target_interest)
        return path
    except nx.NetworkXNoPath:
        return None


def find_matching_positions(current_skills, path, target_role):
    matches = []
    stepping_stones = path[1:-1] if len(path) > 2 else []

    # Positions that would be stepping stones to ideal role
    for j, data in open_positions.items():
        if data["related_to"] in stepping_stones:
            match_pct = compute_match_percentage(current_skills, data["required_skills"])
            matches.append((j, match_pct))

    # Positions directly related to ideal role
    for k, data in open_positions.items():
        # Limit to 3 matches
        if len(matches) >= 3:
            break
        if data["related_to"] == target_role or data["related_to"] in career_graph.nodes:
            match_percentage = compute_match_percentage(current_skills, data["required_skills"])
            # label = "target-related" if data["related_to"] == target_role else "related"
            if (k, match_percentage) not in matches:
                matches.append((k, match_percentage))

    # Sort by percentage to display jobs in order most closely currently aligned with skill-wise.
    matches.sort(key=lambda x: x[1], reverse=True)
    # Limit to 3 matches. Add this just in case I have more than 3 here.
    return matches[:3]


stop_words = set(stopwords.words('english'))


def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
    return keywords


if __name__ == "__main__":
    print("Welcome to your Career Pathway AI Assistant! I'm here to help you plan your career.")
    my_current_role = input("Enter your current role (i.e., Junior Developer): ").strip()
    interests = input("Enter your career interests (i.e., become a Data Scientist or Tech Lead): ").strip()

    # Extract keywords from interests
    my_interests = extract_keywords(interests)
    possible_targets = [node for node in career_graph.nodes if any(interest in node.lower() for interest in my_interests)]
    if not possible_targets:
        print("Sorry, I couldn't match your interests to a known role. Try again.")
        sys.exit(0)
    my_target_role = possible_targets[0]

    my_career_path = plan_career_path(my_current_role, my_target_role)
    if not my_career_path:
        print(f"No direct path from '{my_current_role}' to '{my_target_role}'. Consider related roles.")
        sys.exit(0)

    print(f"\nSuggested Career Path from {my_current_role} to {my_target_role}:")
    my_current_skills = career_graph.nodes[my_current_role]['skills'][:]
    for i in range(len(my_career_path) - 1):
        from_role = my_career_path[i]
        to_role = my_career_path[i + 1]
        required = career_graph[from_role][to_role]['required_skills']
        gaps = identify_skill_gaps(my_current_skills, required)
        print(f"- Transition to {to_role}: Requires {', '.join(required)}. {gaps}")
        my_current_skills.extend(required)

    # Match and list 3 open positions, including stepping stones
    matching_positions = find_matching_positions(career_graph.nodes[my_current_role]['skills'], my_career_path, my_target_role)
    print("\nTop Matching Open Positions based on your current skills and goals, including stepping stones, max 3 positions listed:")
    for position, percentage_match in matching_positions:
        print(f"- {position}: {percentage_match:.2f}% skill match")

    print("\nThis is a personalized plan based on your inputs.")
    print("Talk with your manager to lay out more concrete steps or for additional advice and guidance.")
