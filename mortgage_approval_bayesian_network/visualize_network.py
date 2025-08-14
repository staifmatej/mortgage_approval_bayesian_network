"""Module for creating dynamic Seaborn-based visualizations of the Bayesian Network structure."""
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

def get_all_nodes_from_model(gbn_model=None):
    """Extract all nodes from the GaussianBayesianNetwork model."""
    if gbn_model is None:
        gbn_model = GaussianBayesianNetwork(save_diagram_to_png=False)
    return list(gbn_model.loan_approval_model.nodes())

def get_node_colors(nodes):
    """Dynamically define node categories and their corresponding colors based on node names."""
    node_colors = {}
    
    for node in nodes:
        # Final decision (light blue) - check first
        if node == 'loan_approved':
            node_colors[node] = '#87cefa'
        # Risk assessment (red)
        elif node in ['defaulted', 'credit_history']:
            node_colors[node] = '#e74c3c'
        # Demographics & Employment (blue)
        elif node in ['age', 'government_employee', 'highest_education', 'employment_type', 'len_employment', 'size_of_company']:
            node_colors[node] = '#3498db'
        # Financial inputs (green)
        elif node in ['reported_monthly_income', 'total_existing_debt', 'investments_value', 'property_owned_value', 
                      'housing_status', 'avg_salary']:
            node_colors[node] = '#2ecc71'
        # Loan parameters (orange)
        elif node in ['loan_amount', 'loan_term']:
            node_colors[node] = '#f39c12'
        # Computed variables (purple)
        elif node in ['stability_income', 'total_stable_income_monthly', 'core_net_worth', 'monthly_payment', 'mortgage_end_age', 
                      'years_of_mortgage_after_retirement']:
            node_colors[node] = '#9b59b6'
        # Risk ratios (yellow)
        elif node in ['ratio_income_debt', 'ratio_debt_net_worth', 'ratio_payment_to_income', 'ratio_income_to_avg_salary']:
            node_colors[node] = '#f1c40f'
        # Default color (should not happen with current nodes)
        else:
            node_colors[node] = '#95a5a6'
    
    return node_colors

def create_dynamic_layout(nodes):
    """Create dynamic hierarchical layout based on actual nodes in the network."""
    pos = {}
    
    # Categorize nodes dynamically
    categories = {
        'input': [],
        'financial': [],
        'loan': [],
        'computed': [],
        'ratios': [],
        'decision': []
    }
    
    for node in nodes:
        if node == 'loan_approved':
            categories['decision'].append(node)
        elif node in ['defaulted', 'credit_history']:
            categories['decision'].append(node)
        elif node in ['ratio_income_debt', 'ratio_debt_net_worth', 'ratio_payment_to_income', 'ratio_income_to_avg_salary']:
            categories['ratios'].append(node)
        elif node in ['stability_income', 'total_stable_income_monthly', 'core_net_worth', 'monthly_payment', 'mortgage_end_age', 
                      'years_of_mortgage_after_retirement']:
            categories['computed'].append(node)
        elif node in ['loan_amount', 'loan_term']:
            categories['loan'].append(node)
        elif node in ['reported_monthly_income', 'total_existing_debt', 'investments_value', 'property_owned_value', 
                      'housing_status', 'avg_salary']:
            categories['financial'].append(node)
        else:  # Demographics & Employment
            categories['input'].append(node)
    
    # Position nodes by category
    y_positions = {
        'input': 10,
        'financial': 8,
        'loan': 6,
        'computed': 4,
        'ratios': 2,
        'decision': 0
    }
    
    for category, nodes_in_cat in categories.items():
        if nodes_in_cat:
            x_start = -len(nodes_in_cat) * 2
            for i, node in enumerate(sorted(nodes_in_cat)):
                if category == 'decision' and node == 'loan_approved':
                    pos[node] = (0, -2)
                elif category == 'decision' and node == 'defaulted':
                    pos[node] = (0, 0)
                else:
                    pos[node] = (x_start + i * 4, y_positions[category])
    
    return pos

def calculate_edge_widths(G):
    """Calculate edge widths based on importance of connections."""
    edge_widths = []
    for edge in G.edges():
        if edge[1] == 'loan_approved':
            edge_widths.append(3.0)
        elif edge[1] == 'defaulted':
            edge_widths.append(2.5)
        elif any(keyword in edge[1] for keyword in ['stability_income', 'total_stable']):
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

def get_node_positions(G, nodes):
    """Get node positions using graphviz or custom layout."""
    try:
        return nx.nx_agraph.graphviz_layout(G, prog='dot')
    except (ImportError, AttributeError):
        return create_dynamic_layout(nodes)

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
    
    # Get all nodes dynamically
    all_nodes = get_all_nodes_from_model(gbn_model)
    
    # Create graph and get dynamic colors
    G = create_graph_from_model(gbn_model)
    node_colors = get_node_colors(all_nodes)
    
    plt.figure(figsize=(20, 13))
    ax = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    
    # Get positions dynamically
    pos = get_node_positions(G, all_nodes)
    
    draw_network_edges(G, pos)
    draw_network_nodes(G, pos, node_colors)
    draw_network_labels(G, pos)
    finalize_plot(ax)
    save_visualization()

def save_visualization():
    """Save the visualization to file and manage old files."""
    os.makedirs('diagram_photos', exist_ok=True)
    existing_files = glob.glob('diagram_photos/*interactive*.png')
    if len(existing_files) >= 2:
        existing_files.sort(key=os.path.getmtime)
        for f in existing_files[:-1]:
            os.remove(f)
    plt.savefig('diagram_photos/bayesian_network_seaborn.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Dynamic Bayesian Network visualization saved to diagram_photos/bayesian_network_seaborn.png")

if __name__ == "__main__":
    create_bayesian_network_visualization()