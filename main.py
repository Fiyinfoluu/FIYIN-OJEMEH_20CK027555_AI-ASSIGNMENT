import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import seaborn as sns

# Set up visualization styles
plt.style.use('dark_background')
sns.set(style="darkgrid")
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

def create_synthetic_data(seed=42):
    """Generate synthetic data: avg_pressure → daily_output"""
    np.random.seed(seed)
    X_synthetic = np.linspace(400, 500, 50).reshape(-1, 1)  # avg_pressure in psi
    y_synthetic = 2.5 * X_synthetic.flatten() + 50 + np.random.normal(0, 20, 50)  # daily_output (barrels)
    
    # Normalize the feature
    scaler = StandardScaler()
    X_synthetic_normalized = scaler.fit_transform(X_synthetic)
    
    print("Synthetic Design Matrix (first 5 rows):\n", X_synthetic_normalized[:5])
    print("Synthetic Target Vector (first 5 values):\n", y_synthetic[:5])
    
    # Save synthetic data
    df_synth = pd.DataFrame({
        "avg_pressure": X_synthetic.flatten(),
        "avg_pressure_normalized": X_synthetic_normalized.flatten(),
        "daily_output": y_synthetic
    })
    df_synth.to_csv("synthetic_oil_data.csv", index=False)
    
    return X_synthetic, X_synthetic_normalized, y_synthetic, scaler

def load_real_data(file_path):
    """Load and preprocess real-world oil production data"""
    try:
        df = pd.read_csv(file_path)
        
        # Remove 'Total' row and strip spaces
        df = df.drop(df[df["Year/Month"].str.contains("Total", na=False)].index)
        df.columns = [col.strip() for col in df.columns]
        df["Year/Month"] = df["Year/Month"].str.strip()
        
        # Melt the data from wide to long format
        df_melted = df.melt(id_vars="Year/Month", var_name="Year", value_name="Production")
        
        # Clean and convert
        df_melted["Production"] = df_melted["Production"].replace({",": ""}, regex=True).astype(float)
        df_melted["Year"] = df_melted["Year"].astype(int)
        
        # Map months to numbers
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        df_melted["Month"] = df_melted["Year/Month"].map(month_map)
        
        # Construct design matrix and target
        X_real = df_melted[["Year", "Month"]].values
        y_real = df_melted["Production"].values
        
        # Normalize features
        scaler = StandardScaler()
        X_real_normalized = scaler.fit_transform(X_real)
        
        # Display
        print("\nReal Oil Design Matrix (first 5 rows):\n", X_real_normalized[:5])
        print("Real Target Vector (first 5 values):\n", y_real[:5])
        
        # Save normalized real data
        df_real = pd.DataFrame({
            "year": X_real[:, 0],
            "month": X_real[:, 1],
            "year_normalized": X_real_normalized[:, 0],
            "month_normalized": X_real_normalized[:, 1],
            "production": y_real
        })
        df_real.to_csv("normalized_real_oil_data.csv", index=False)
        
        return X_real, X_real_normalized, y_real, df_melted
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        print("Creating a sample real data file for demonstration...")
        
        # Create sample data if file not found
        years = np.repeat(np.arange(2004, 2014), 12)
        months = np.tile(np.arange(1, 13), 10)
        production = 1000 + 50 * np.sin(months/2) + years * 10 + np.random.normal(0, 100, 120)
        
        # Create a DataFrame
        sample_data = pd.DataFrame({
            "Year": years,
            "Month": months,
            "Production": production
        })
        
        # Save the sample data
        sample_data.to_csv("sample_nnpc_crude_oil_production.csv", index=False)
        
        # Construct design matrix and target
        X_real = sample_data[["Year", "Month"]].values
        y_real = sample_data["Production"].values
        
        # Normalize features
        scaler = StandardScaler()
        X_real_normalized = scaler.fit_transform(X_real)
        
        print("\nCreated sample data. Using this for demonstration.")
        print("Real Oil Design Matrix (first 5 rows):\n", X_real_normalized[:5])
        print("Real Target Vector (first 5 values):\n", y_real[:5])
        
        return X_real, X_real_normalized, y_real, sample_data

def compute_cost_functions(X, y, params):
    """Calculate cost functions for different parameter sets"""
    results = []
    
    for w, b in params:
        if X.ndim == 2 and X.shape[1] == 1:  # Single feature
            y_pred = X.flatten() * w + b
        elif X.ndim == 2 and X.shape[1] > 1:  # Multiple features
            y_pred = np.dot(X, np.array([w] * X.shape[1])) + b
        else:  # Handle flattened input
            y_pred = X * w + b
            
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        results.append((w, b, mse, mae))
    
    return results

def gradient_descent_single_feature(X, y, learning_rate=0.01, epochs=100, verbose=True):
    """Implement gradient descent for a single feature"""
    # Initialize parameters
    w = 0
    b = 0
    m = len(X)
    history = {'loss': [], 'w': [], 'b': []}
    
    # Ensure X is flattened
    X_flat = X.flatten() if X.ndim > 1 and X.shape[1] == 1 else X
    
    # Gradient Descent Loop
    for epoch in range(epochs):
        # Forward pass
        y_pred = X_flat * w + b
        
        # Compute loss
        loss = np.mean((y_pred - y) ** 2)
        
        # Compute gradients
        dw = (2/m) * np.sum((y_pred - y) * X_flat)
        db = (2/m) * np.sum(y_pred - y)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Store history
        history['loss'].append(loss)
        history['w'].append(w)
        history['b'].append(b)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: w={w:.4f}, b={b:.4f}, loss={loss:.4f}")
    
    return w, b, history

def gradient_descent_multi_feature(X, y, learning_rate=0.01, epochs=100, verbose=True):
    """Implement gradient descent for multiple features"""
    # Get number of features and samples
    m, n = X.shape
    
    # Initialize weights and bias
    w = np.zeros(n)
    b = 0
    history = {'loss': [], 'w': [], 'b': []}
    
    # Gradient Descent Loop
    for epoch in range(epochs):
        # Forward pass
        y_pred = np.dot(X, w) + b
        
        # Compute loss
        loss = np.mean((y_pred - y) ** 2)
        
        # Compute gradients
        dw = (2/m) * np.dot(X.T, (y_pred - y))
        db = (2/m) * np.sum(y_pred - y)
        
        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Store history
        history['loss'].append(loss)
        history['w'].append(w.copy())
        history['b'].append(b)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch}: loss={loss:.4f}")
    
    return w, b, history

def plot_fitted_line(X, y, w, b, title, xlabel, ylabel):
    """Plot data points and fitted line"""
    plt.figure(figsize=(12, 6))
    
    # Plot actual data points
    plt.scatter(X, y, color=colors[0], alpha=0.6, label="Actual Data")
    
    # Sort X for line plot
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx].flatten()
    
    # Calculate predictions
    if isinstance(w, np.ndarray) and len(w) > 1:
        y_pred = np.dot(X, w) + b
        y_pred_sorted = y_pred[sort_idx]
    else:
        y_pred = X.flatten() * w + b
        y_pred_sorted = y_pred[sort_idx]
    
    # Plot fitted line
    plt.plot(X_sorted, y_pred_sorted, color=colors[1], linewidth=2, label="Fitted Line")
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add equation of the line
    if isinstance(w, np.ndarray) and len(w) > 1:
        equation = f"y = {w[0]:.4f} * x1 + {w[1]:.4f} * x2 + {b:.4f}"
    else:
        equation = f"y = {w:.4f} * x + {b:.4f}"
    
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                fontsize=12, backgroundcolor='black', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_loss_history(history, title="Loss Over Iterations"):
    """Plot the loss history during training"""
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], color=colors[2], linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluate model performance with multiple metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def main():
    # Part 1: Generate synthetic data
    print("\n=== Activity 1: Data Preparation ===")
    X_synthetic, X_synthetic_normalized, y_synthetic, _ = create_synthetic_data()
    
    # Part 2: Calculate cost functions
    print("\n=== Activity 2: Calculating Cost Functions ===")
    params = [(1.5, 0.5), (2.0, 10), (2.5, 50)]
    results = compute_cost_functions(X_synthetic_normalized, y_synthetic, params)
    
    print("\n--- MSE and MAE for different parameters ---")
    for w, b, mse, mae in results:
        print(f"(w={w}, b={b}) -> MSE: {mse:.2f}, MAE: {mae:.2f}")
    
    # Part 3: Gradient Descent on Synthetic Data
    print("\n=== Activity 3: Implementing Gradient Descent ===")
    w_synth, b_synth, history_synth = gradient_descent_single_feature(
        X_synthetic_normalized, y_synthetic, learning_rate=0.01, epochs=100
    )
    
    # Plot loss over iterations
    plot_loss_history(history_synth, "Synthetic Data - Loss Over Iterations")
    
    # Generate predictions
    y_pred_synth = X_synthetic_normalized.flatten() * w_synth + b_synth
    
    # Evaluate model
    synth_metrics = evaluate_model(y_synthetic, y_pred_synth, "Synthetic Data Model")
    
    # Plot fitted line
    plot_fitted_line(
        X_synthetic, y_synthetic, w_synth, b_synth,
        "Synthetic Data: Pressure vs. Output",
        "Average Pressure (psi)", "Daily Output (barrels)"
    )
    
    # Part 4: Real-world Data Analysis
    print("\n=== Activity 4: Applying to Real Data ===")
    
    # Try to load the provided file, use a default path, or create sample data if not found
    try:
        file_path = "nnpc-crude-oil-production-2004-2013.csv"
        X_real, X_real_normalized, y_real, df_real = load_real_data(file_path)
    except:
        # If file not found, try alternate locations or create sample data
        possible_paths = [
            "data/nnpc-crude-oil-production-2004-2013.csv",
            "nnpc-crude-oil-production.csv",
            "oil_production_data.csv"
        ]
        
        for path in possible_paths:
            try:
                X_real, X_real_normalized, y_real, df_real = load_real_data(path)
                break
            except:
                continue
        else:
            # If all attempts fail, use the function's sample data generation
            X_real, X_real_normalized, y_real, df_real = load_real_data("nonexistent_file.csv")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_real_normalized, y_real, test_size=0.2, random_state=42
    )
    
    # Apply gradient descent for multiple features
    w_real, b_real, history_real = gradient_descent_multi_feature(
        X_train, y_train, learning_rate=0.01, epochs=100
    )
    
    # Plot loss over iterations
    plot_loss_history(history_real, "Real Data - Loss Over Iterations")
    
    # Generate predictions
    y_pred_test = np.dot(X_test, w_real) + b_real
    
    # Evaluate model
    real_metrics = evaluate_model(y_test, y_pred_test, "Real Data Model")
    
    # Create a time series plot for real data if we have a time-based dataset
    if isinstance(df_real, pd.DataFrame) and 'Year' in df_real.columns:
        plt.figure(figsize=(14, 7))
        
        # Sort by date if possible
        if 'Month' in df_real.columns:
            df_real['Date'] = pd.to_datetime(df_real[['Year', 'Month']].assign(Day=1))
            df_real = df_real.sort_values('Date')
            
            # Plot actual values
            plt.plot(df_real['Date'], df_real['Production'], 
                    color=colors[0], label='Actual Production')
            
            # Make predictions on the entire dataset
            X_all_norm = StandardScaler().fit_transform(df_real[['Year', 'Month']].values)
            y_pred_all = np.dot(X_all_norm, w_real) + b_real
            
            # Plot predictions
            plt.plot(df_real['Date'], y_pred_all, 
                    color=colors[1], label='Predicted Production', linestyle='--')
            
            plt.title('Oil Production Over Time: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Production')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("oil_production_time_series.png")
            plt.show()
    
    # Summary of both models
    print("\n=== Model Comparison Summary ===")
    print(f"Synthetic Data Model: R² = {synth_metrics['r2']:.4f}, RMSE = {synth_metrics['rmse']:.2f}")
    print(f"Real Data Model: R² = {real_metrics['r2']:.4f}, RMSE = {real_metrics['rmse']:.2f}")
    print("\nAnalysis complete! Visualization files have been saved.")

if __name__ == "__main__":
    main()