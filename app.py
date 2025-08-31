# Sales Performance Dashboard
# Complete Python application with Streamlit
# Author: Nenavath Sandeep
# GitHub: https://github.com/Sandeep1-prog

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Sales Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_sample_data():
    """Generate realistic sales data for demonstration"""
    np.random.seed(42)
    
    # Date range
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Products
    products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Camera', 
               'Watch', 'Speaker', 'Monitor', 'Keyboard', 'Mouse']
    
    # Regions
    regions = ['North America', 'Europe', 'Asia Pacific', 'South America', 'Africa']
    
    # Sales representatives
    sales_reps = ['John Smith', 'Sarah Johnson', 'Mike Davis', 'Emily Brown', 
                 'David Wilson', 'Lisa Anderson', 'Tom Garcia', 'Anna Martinez']
    
    # Generate data
    data = []
    for _ in range(10000):  # 10,000+ records as mentioned in CV
        date = np.random.choice(date_range)
        product = np.random.choice(products)
        region = np.random.choice(regions)
        sales_rep = np.random.choice(sales_reps)
        
        # Seasonal patterns
        month = date.month
        if month in [11, 12, 1]:  # Holiday season
            seasonal_multiplier = 1.5
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.2
        else:
            seasonal_multiplier = 1.0
        
        # Base price by product
        base_prices = {'Laptop': 800, 'Smartphone': 600, 'Tablet': 300, 
                      'Headphones': 150, 'Camera': 500, 'Watch': 200,
                      'Speaker': 100, 'Monitor': 400, 'Keyboard': 80, 'Mouse': 50}
        
        unit_price = base_prices[product] * (1 + np.random.normal(0, 0.1))
        quantity = max(1, int(np.random.poisson(3) * seasonal_multiplier))
        total_sales = unit_price * quantity
        
        # Add some profit margin calculation
        cost_ratio = np.random.uniform(0.6, 0.8)
        profit = total_sales * (1 - cost_ratio)
        
        data.append({
            'Date': date,
            'Product': product,
            'Region': region,
            'Sales_Rep': sales_rep,
            'Unit_Price': round(unit_price, 2),
            'Quantity': quantity,
            'Total_Sales': round(total_sales, 2),
            'Profit': round(profit, 2),
            'Month': date.strftime('%B'),
            'Year': date.year,
            'Quarter': f"Q{((date.month-1)//3)+1}"
        })
    
    return pd.DataFrame(data)

@st.cache_data
def calculate_metrics(df, filtered_df):
    """Calculate key performance indicators"""
    total_sales = filtered_df['Total_Sales'].sum()
    total_profit = filtered_df['Profit'].sum()
    total_orders = len(filtered_df)
    avg_order_value = filtered_df['Total_Sales'].mean()
    
    # Growth calculations (comparing to previous period)
    if len(df) > len(filtered_df):
        # Calculate growth rate
        previous_sales = df['Total_Sales'].sum() - total_sales
        growth_rate = ((total_sales - previous_sales) / previous_sales) * 100 if previous_sales > 0 else 0
    else:
        growth_rate = 0
    
    profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'avg_order_value': avg_order_value,
        'growth_rate': growth_rate,
        'profit_margin': profit_margin
    }

def create_time_series_chart(df):
    """Create time series chart for sales trends"""
    daily_sales = df.groupby('Date')['Total_Sales'].sum().reset_index()
    
    fig = px.line(daily_sales, x='Date', y='Total_Sales',
                  title='Daily Sales Trend',
                  labels={'Total_Sales': 'Total Sales ($)', 'Date': 'Date'})
    fig.update_layout(height=400)
    return fig

def create_regional_analysis(df):
    """Create regional sales analysis"""
    regional_data = df.groupby('Region').agg({
        'Total_Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    fig = px.bar(regional_data, x='Region', y='Total_Sales',
                 title='Sales by Region',
                 labels={'Total_Sales': 'Total Sales ($)'})
    fig.update_layout(height=400)
    return fig

def create_product_performance(df):
    """Create product performance analysis"""
    product_data = df.groupby('Product').agg({
        'Total_Sales': 'sum',
        'Quantity': 'sum',
        'Profit': 'sum'
    }).reset_index().sort_values('Total_Sales', ascending=False)
    
    fig = px.bar(product_data.head(10), x='Product', y='Total_Sales',
                 title='Top 10 Products by Sales',
                 labels={'Total_Sales': 'Total Sales ($)'})
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def create_seasonal_pattern(df):
    """Create seasonal pattern analysis"""
    monthly_data = df.groupby(['Month', 'Year'])['Total_Sales'].sum().reset_index()
    
    # Create a proper month order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=month_order, ordered=True)
    monthly_data = monthly_data.sort_values(['Year', 'Month'])
    
    fig = px.line(monthly_data, x='Month', y='Total_Sales', color='Year',
                  title='Seasonal Sales Patterns',
                  labels={'Total_Sales': 'Total Sales ($)'})
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def create_sales_rep_performance(df):
    """Create sales representative performance analysis"""
    rep_data = df.groupby('Sales_Rep').agg({
        'Total_Sales': 'sum',
        'Profit': 'sum',
        'Total_Sales': 'count'
    }).reset_index()
    rep_data.columns = ['Sales_Rep', 'Total_Sales', 'Total_Profit', 'Orders_Count']
    rep_data = rep_data.sort_values('Total_Sales', ascending=True)
    
    fig = px.bar(rep_data, x='Total_Sales', y='Sales_Rep',
                 orientation='h',
                 title='Sales Representative Performance',
                 labels={'Total_Sales': 'Total Sales ($)', 'Sales_Rep': 'Sales Representative'})
    fig.update_layout(height=400)
    return fig

# Main application
def main():
    # Title
    st.markdown('<h1 class="main-header">📊 Sales Performance Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading sales data...'):
        df = generate_sample_data()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Date filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    
    # Product filter
    products = st.sidebar.multiselect(
        "Select Products",
        options=df['Product'].unique(),
        default=df['Product'].unique()
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique()
    )
    
    # Sales Rep filter
    sales_reps = st.sidebar.multiselect(
        "Select Sales Representatives",
        options=df['Sales_Rep'].unique(),
        default=df['Sales_Rep'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1]) &
        (df['Product'].isin(products)) &
        (df['Region'].isin(regions)) &
        (df['Sales_Rep'].isin(sales_reps))
    ]
    
    # Calculate metrics
    metrics = calculate_metrics(df, filtered_df)
    
    # Display key metrics
    st.subheader("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${metrics['total_sales']:,.2f}",
            delta=f"{metrics['growth_rate']:.1f}%" if metrics['growth_rate'] != 0 else None
        )
    
    with col2:
        st.metric(
            label="Total Profit",
            value=f"${metrics['total_profit']:,.2f}",
            delta=f"{metrics['profit_margin']:.1f}% margin"
        )
    
    with col3:
        st.metric(
            label="Total Orders",
            value=f"{metrics['total_orders']:,}"
        )
    
    with col4:
        st.metric(
            label="Average Order Value",
            value=f"${metrics['avg_order_value']:,.2f}"
        )
    
    # Charts section
    st.subheader("📊 Sales Analytics")
    
    # Time series chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_time_series_chart(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_regional_analysis(filtered_df), use_container_width=True)
    
    # Product and seasonal analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_product_performance(filtered_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_seasonal_pattern(filtered_df), use_container_width=True)
    
    # Sales rep performance
    st.plotly_chart(create_sales_rep_performance(filtered_df), use_container_width=True)
    
    # Data table
    st.subheader("📋 Raw Data")
    st.dataframe(
        filtered_df.sort_values('Date', ascending=False).head(1000),
        use_container_width=True
    )
    
    # Export functionality
    st.subheader("💾 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.info(f"Dataset contains {len(filtered_df):,} records")
    
    # Footer
    st.markdown("---")
    st.markdown("**Developed by:** Nenavath Sandeep | **GitHub:** [Sandeep1-prog](https://github.com/Sandeep1-prog)")

if __name__ == "__main__":
    main()
