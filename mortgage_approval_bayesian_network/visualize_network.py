"""Module for creating Seaborn-based visualizations of the Bayesian Network structure."""
import os
import glob
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gaussian_bayesian_network import GaussianBayesianNetwork
def get_edges_from_gbn_model(gbn_model=None):
    """Extract edges from the GaussianBayesianNetwork model."""
    if gbn_model is None:
        gbn_model = GaussianBayesianNetwork(save_diagram_to_png=False)
    return list(gbn_model.loan_approval_model.edges())
def get_node_colors():
    """Define node categories and their corresponding colors."""
    return {
        'government_employee': '#3498db',
        'age_young': '#3498db',
        'age_prime': '#3498db',
        'age_senior': '#3498db',
        'age_old': '#3498db',
        'highest_education': '#3498db',
        'employment_type': '#3498db',
        'len_employment': '#3498db',
        'size_of_company': '#3498db',
        'housing_status': '#3498db',
        'reported_monthly_income': '#2ecc71',
        'total_existing_debt': '#2ecc71',
        'investments_value': '#2ecc71',
        'property_owned_value': '#2ecc71',
        'avg_salary': '#2ecc71',
        'credit_history': '#2ecc71',
        'loan_amount': '#f39c12',
        'loan_term': '#f39c12',
        'stability_income': '#9b59b6',
        'total_stable_income_monthly': '#9b59b6',
        'core_net_worth': '#9b59b6',
        'monthly_payment': '#9b59b6',
        'ratio_income_debt': '#f1c40f',
        'ratio_debt_net_worth': '#f1c40f',
        'ratio_payment_to_income': '#f1c40f',
        'ratio_income_to_avg_salary': '#f1c40f',
        'defaulted': '#e74c3c',
        'loan_approved': '#87cefa'
    }
def create_custom_layout():
    """Create custom hierarchical layout for the network nodes."""
    pos = {}
    layer1 = ['government_employee', 'age_young', 'age_prime', 'age_senior', 'age_old',
              'highest_education', 'employment_type', 'len_employment', 'size_of_company']
    for i, node in enumerate(layer1):
        pos[node] = (i * 2 - 8, 10)
    layer2 = ['reported_monthly_income', 'total_existing_debt', 'investments_value',
              'property_owned_value', 'avg_salary', 'housing_status', 'credit_history']
    for i, node in enumerate(layer2):
        pos[node] = (i * 2.8 - 8, 8)
    layer3 = ['loan_amount', 'loan_term']
    for i, node in enumerate(layer3):
        pos[node] = (i * 4 - 2, 6)
    layer4 = ['stability_income', 'total_stable_income_monthly', 'core_net_worth', 'monthly_payment']
    for i, node in enumerate(layer4):
        pos[node] = (i * 4 - 6, 4)
    layer5 = ['ratio_income_debt', 'ratio_debt_net_worth', 'ratio_payment_to_income', 'ratio_income_to_avg_salary']
    for i, node in enumerate(layer5):
        pos[node] = (i * 4 - 6, 2)
    pos['defaulted'] = (0, 0)
    pos['loan_approved'] = (0, -2)
    return pos
def calculate_edge_widths(G):
    """Calculate edge widths based on importance of connections."""
    edge_widths = []
    for edge in G.edges():
        if edge[1] == 'loan_approved':
            edge_widths.append(3.0)
        elif edge[1] == 'defaulted':
            edge_widths.append(2.5)
        elif edge[1] in ['stability_income', 'total_stable_income_monthly']:
            edge_widths.append(2.0)
        else:
            edge_widths.append(1.5)
    return edge_widths
def create_legend_elements():
    """Create legend elements for the visualization."""
    return [
        mpatches.Patch(color='#3498db', label='Employment/Demographics'),
        mpatches.Patch(color='#2ecc71', label='Financial Inputs'),
        mpatches.Patch(color='#f39c12', label='Loan Parameters'),
        mpatches.Patch(color='#9b59b6', label='Computed Variables'),
        mpatches.Patch(color='#f1c40f', label='Risk Ratios'),
        mpatches.Patch(color='#e74c3c', label='Risk Assessment'),
        mpatches.Patch(color='#87cefa', label='Final Decision')
    ]
def setup_plot_style():
    """Set up the plot style and configuration."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (20, 16)
    plt.rcParams['font.size'] = 10
def create_graph_from_model(gbn_model):
    """Create a directed graph from the Bayesian Network model."""
    edges = get_edges_from_gbn_model(gbn_model)
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G
def get_node_positions(G):
    """Get node positions using graphviz or custom layout."""
    try:
        return nx.nx_agraph.graphviz_layout(G, prog='dot')
    except (ImportError, AttributeError):
        return create_custom_layout()
def draw_network_edges(G, pos):
    """Draw the network edges with varying thickness."""
    edge_widths = calculate_edge_widths(G)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5,
                          width=edge_widths, arrows=True,
                          arrowsize=20, arrowstyle='->', node_size=3000)
def draw_network_nodes(G, pos, node_colors):
    """Draw the network nodes with appropriate colors."""
    for node in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                              node_color=node_colors.get(node, '#c0392b'),
                              node_size=3000, alpha=0.9)
def draw_network_labels(G, pos):
    """Draw labels for network nodes."""
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
def finalize_plot(ax):
    """Add legend, title, and finalize the plot."""
    legend_elements = create_legend_elements()
    ax.legend(handles=legend_elements, loc='lower left', fontsize=14)
    plt.title('\nDiagram of Gaussian Bayesian Network for Mortgage Approval\n',
              fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
def create_bayesian_network_visualization(gbn_model=None):
    """Create a beautiful visualization of the Bayesian Network using Seaborn with hierarchical layout."""
    setup_plot_style()
    G = create_graph_from_model(gbn_model)
    node_colors = get_node_colors()
    plt.figure(figsize=(20, 13))
    ax = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    pos = get_node_positions(G)
    draw_network_edges(G, pos)
    draw_network_nodes(G, pos, node_colors)
    draw_network_labels(G, pos)
    finalize_plot(ax)
    save_visualization()
def save_visualization():
    """Save the visualization to file and manage old files."""
    os.makedirs('diagram_photos', exist_ok=True)
    existing_files = glob.glob('diagram_photos/*.png')
    if len(existing_files) >= 2:
        existing_files.sort(key=os.path.getmtime)
        for f in existing_files[:-1]:
            os.remove(f)
    plt.savefig('diagram_photos/bayesian_network_seaborn.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Bayesian Network visualization saved to diagram_photos/bayesian_network_seaborn.png")

if __name__ == "__main__":
    create_bayesian_network_visualization()
