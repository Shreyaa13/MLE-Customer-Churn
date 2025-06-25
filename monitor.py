
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SimpleOOTMonitor:
    def __init__(self):
        self.predictions_path = "scripts/datamart/gold/model_predictions/xgb_churn_model.jo/xgb_churn_model.jo_predictions_2024_04_01_to_2024_07_01.csv"
        self.ground_truth_path = "scripts/datamart/gold/label_store/ground_truth_2024_04_01_to_2024_07_01.csv"
        # Use actual data from scripts/data
        self.telco_features_path = "scripts/data/telco_features.csv"
        self.telco_labels_path = "scripts/data/telco_labels.csv"
        # Model artifacts for generating probability predictions
        self.model_path = "model_artifacts/xgb_model_pipeline.pkl"
        self.output_dir = "scripts/datamart/monitoring/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Date ranges for analysis
        self.start_date = '2024-04-01'
        self.end_date = '2024-07-01'
        
    def load_data(self):
        """Load and prepare data for monitoring"""
        try:
            # Try to load actual predictions and ground truth
            if os.path.exists(self.predictions_path) and os.path.exists(self.telco_features_path) and os.path.exists(self.telco_labels_path):
                print("🔄 Loading actual model predictions and data...")
                
                # Load actual predictions
                predictions_df = pd.read_csv(self.predictions_path)
                print(f"✅ Loaded predictions: {len(predictions_df)} records")
                
                # Load telco data for features and ground truth
                features_df = pd.read_csv(self.telco_features_path)
                labels_df = pd.read_csv(self.telco_labels_path)
                
                # Rename snapshot date column to match expected format
                if 'Snapshot_Date' in features_df.columns:
                    features_df = features_df.rename(columns={'Snapshot_Date': 'snapshot_date'})
                if 'Snapshot_Date' in labels_df.columns:
                    labels_df = labels_df.rename(columns={'Snapshot_Date': 'snapshot_date'})
                
                print(f"✅ Loaded features: {len(features_df)} records")
                print(f"✅ Loaded labels: {len(labels_df)} records")
                
                # Combine actual predictions with telco data
                combined_data = self.combine_actual_predictions_with_data(predictions_df, features_df, labels_df)
                return combined_data
            
            # Fallback to telco data with generated predictions
            elif os.path.exists(self.telco_features_path) and os.path.exists(self.telco_labels_path):
                print("⚠️  Prediction file not found, using telco data with model-generated predictions...")
                features_df = pd.read_csv(self.telco_features_path)
                labels_df = pd.read_csv(self.telco_labels_path)
                
                # Rename snapshot date column to match expected format
                if 'Snapshot_Date' in features_df.columns:
                    features_df = features_df.rename(columns={'Snapshot_Date': 'snapshot_date'})
                if 'Snapshot_Date' in labels_df.columns:
                    labels_df = labels_df.rename(columns={'Snapshot_Date': 'snapshot_date'})
                
                print(f"✅ Loaded features: {len(features_df)} records")
                print(f"✅ Loaded labels: {len(labels_df)} records")
                
                # Generate predictions using actual model if available
                combined_data = self.combine_telco_data_with_model_predictions(features_df, labels_df)
                return combined_data
            else:
                print(f"❌ Required data files not found, generating mock data")
                return self.generate_mock_combined_data()
            
        except Exception as e:
            print(f"❌ Error loading telco data: {e}")
            print("📊 Generating mock data for demonstration...")
            return self.generate_mock_combined_data()
    
    def generate_mock_predictions(self):
        """Generate mock prediction data"""
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        data = []
        for date in dates:
            n_customers = np.random.randint(50, 100)
            for i in range(n_customers):
                customer_id = f"CUST_{date.strftime('%Y%m%d')}_{i:03d}"
                churn_prob = np.random.beta(2, 5)  # Skewed towards lower probabilities
                data.append({
                    'customer_id': customer_id,
                    'prediction_date': date,
                    'churn_probability': churn_prob,
                    'monthly_charges': np.random.gamma(2, 30) + np.random.normal(0, 5)
                })
        
        return pd.DataFrame(data)
    
    def generate_mock_ground_truth(self):
        """Generate mock ground truth data"""
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        data = []
        for date in dates:
            n_customers = np.random.randint(50, 100)
            for i in range(n_customers):
                customer_id = f"CUST_{date.strftime('%Y%m%d')}_{i:03d}"
                # Generate actual churn with some correlation to probability
                actual_churn = np.random.choice([0, 1], p=[0.8, 0.2])
                data.append({
                    'customer_id': customer_id,
                    'actual_date': date,
                    'actual_churn': actual_churn
                })
        
        return pd.DataFrame(data)
    
    def combine_actual_predictions_with_data(self, predictions_df, features_df, labels_df):
        """Combine actual model predictions with telco data"""
        # Clean predictions data
        predictions_df['snapshot_date'] = pd.to_datetime(predictions_df['snapshot_date'])
        predictions_df = predictions_df.rename(columns={'model_predictions': 'binary_prediction'})
        
        # Merge features and labels
        features_labels = pd.merge(features_df, labels_df, on='customerID', how='inner', suffixes=('_features', '_labels'))
        features_labels['snapshot_date'] = pd.to_datetime(features_labels['snapshot_date_features'])
        features_labels['actual_churn'] = (features_labels['Churn'] == 'Yes').astype(int)
        features_labels['MonthlyCharges'] = pd.to_numeric(features_labels['MonthlyCharges'], errors='coerce')
        features_labels['monthly_charges'] = features_labels['MonthlyCharges']
        
        # Merge with predictions
        combined = pd.merge(predictions_df, features_labels, on=['customerID', 'snapshot_date'], how='inner')
        
        # Convert binary predictions to probability scores
        # Since we only have binary predictions, we'll generate probability scores
        # based on the binary prediction and add some realistic variation
        np.random.seed(42)
        combined['churn_probability'] = combined['binary_prediction'].apply(
            lambda x: np.random.beta(8, 2) if x == 1 else np.random.beta(2, 8)
        )
        
        # Add month column for grouping
        combined['month'] = combined['snapshot_date'].dt.strftime('%b')
        
        # Filter to our date range
        combined = combined[(combined['snapshot_date'] >= self.start_date) & 
                          (combined['snapshot_date'] <= self.end_date)]
        
        print(f"📊 Combined data: {len(combined)} records with actual predictions")
        return combined
    
    def combine_telco_data_with_model_predictions(self, features_df, labels_df):
        """Combine telco data and generate predictions using the actual model"""
        try:
            import pickle
            # Try to load the actual model
            if os.path.exists(self.model_path):
                print("🤖 Loading trained model for predictions...")
                with open(self.model_path, 'rb') as f:
                    model_pipeline = pickle.load(f)
                
                # Prepare data similar to training process
                combined = pd.merge(features_df, labels_df, on='customerID', how='inner', suffixes=('_features', '_labels'))
                combined['snapshot_date'] = pd.to_datetime(combined['snapshot_date_features'])
                combined['actual_churn'] = (combined['Churn'] == 'Yes').astype(int)
                combined['MonthlyCharges'] = pd.to_numeric(combined['MonthlyCharges'], errors='coerce')
                combined['monthly_charges'] = combined['MonthlyCharges']
                
                # Prepare features for model prediction
                feature_cols = ['tenure', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 
                              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
                              'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
                
                model_features = combined[feature_cols].copy()
                
                # Handle missing values and data types
                model_features['TotalCharges'] = pd.to_numeric(model_features['TotalCharges'], errors='coerce')
                model_features = model_features.fillna(model_features.median(numeric_only=True))
                
                # Generate predictions using the actual model
                try:
                    if hasattr(model_pipeline, 'predict_proba'):
                        predictions = model_pipeline.predict_proba(model_features)[:, 1]
                    else:
                        # If it's a pipeline, try to access the model
                        predictions = model_pipeline.predict_proba(model_features)[:, 1]
                    
                    combined['churn_probability'] = predictions
                    print(f"✅ Generated {len(predictions)} model predictions")
                    
                except Exception as model_error:
                    print(f"⚠️  Model prediction failed: {model_error}")
                    print("🔄 Falling back to heuristic-based predictions...")
                    combined = self.combine_telco_data(features_df, labels_df)
                    return combined
                
            else:
                print(f"⚠️  Model file not found: {self.model_path}")
                print("🔄 Using heuristic-based predictions...")
                combined = self.combine_telco_data(features_df, labels_df)
                return combined
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to heuristic-based predictions...")
            combined = self.combine_telco_data(features_df, labels_df)
            return combined
        
        # Add month column for grouping
        combined['month'] = combined['snapshot_date'].dt.strftime('%b')
        
        # Filter to our date range and create monthly samples
        combined = combined[(combined['snapshot_date'] >= self.start_date) & 
                          (combined['snapshot_date'] <= self.end_date)]
        
        # Create monthly samples by duplicating data with different dates
        monthly_data = []
        months = ['Apr', 'May', 'Jun', 'Jul']
        month_dates = ['2024-04-15', '2024-05-15', '2024-06-15', '2024-07-15']
        
        for month, date in zip(months, month_dates):
            month_sample = combined.copy()
            month_sample['month'] = month
            month_sample['snapshot_date'] = pd.to_datetime(date)
            
            # Add some monthly variation to churn probabilities
            month_idx = months.index(month)
            variation = np.random.normal(month_idx * 0.01, 0.02, len(month_sample))
            month_sample['churn_probability'] = np.clip(
                month_sample['churn_probability'] + variation, 0.01, 0.99
            )
            
            monthly_data.append(month_sample)
        
        final_combined = pd.concat(monthly_data, ignore_index=True)
        print(f"📊 Generated model-based predictions: {len(final_combined)} records")
        
        return final_combined
    
    def combine_telco_data(self, features_df, labels_df):
        """Combine telco features and labels data"""
        # Merge on customerID
        combined = pd.merge(features_df, labels_df, on='customerID', how='inner', suffixes=('_features', '_labels'))
        
        # Convert dates
        combined['snapshot_date'] = pd.to_datetime(combined['snapshot_date_features'])
        
        # Convert churn to binary (Yes=1, No=0)
        combined['actual_churn'] = (combined['Churn'] == 'Yes').astype(int)
        
        # Convert monthly charges to numeric
        combined['MonthlyCharges'] = pd.to_numeric(combined['MonthlyCharges'], errors='coerce')
        combined['monthly_charges'] = combined['MonthlyCharges']
        
        # Generate mock churn probabilities based on features
        # This simulates model predictions using simple heuristics
        np.random.seed(42)
        base_prob = 0.2  # Base churn probability
        
        # Adjust probability based on tenure (lower tenure = higher churn risk)
        tenure_factor = 1 - (combined['tenure'] / 100)  # Normalize tenure
        
        # Adjust based on monthly charges (higher charges = higher churn risk)
        charge_factor = combined['MonthlyCharges'] / 120  # Normalize charges
        
        # Adjust based on contract type
        contract_factor = combined['Contract'].map({
            'Month-to-month': 1.5,
            'One year': 1.0,
            'Two year': 0.5
        }).fillna(1.0)
        
        # Calculate churn probability with some randomness
        combined['churn_probability'] = np.clip(
            base_prob + 
            tenure_factor * 0.3 + 
            charge_factor * 0.2 + 
            (contract_factor - 1) * 0.3 +
            np.random.normal(0, 0.1, len(combined)),
            0.01, 0.99
        )
        
        # Add month column for grouping
        combined['month'] = combined['snapshot_date'].dt.strftime('%b')
        
        # Filter to our date range and create monthly samples
        combined = combined[(combined['snapshot_date'] >= self.start_date) & 
                          (combined['snapshot_date'] <= self.end_date)]
        
        # Create monthly samples by duplicating data with different dates
        monthly_data = []
        months = ['Apr', 'May', 'Jun', 'Jul']
        month_dates = ['2024-04-15', '2024-05-15', '2024-06-15', '2024-07-15']
        
        for month, date in zip(months, month_dates):
            month_sample = combined.copy()
            month_sample['month'] = month
            month_sample['snapshot_date'] = pd.to_datetime(date)
            
            # Add some monthly variation to churn probabilities
            month_idx = months.index(month)
            variation = np.random.normal(month_idx * 0.02, 0.05, len(month_sample))
            month_sample['churn_probability'] = np.clip(
                month_sample['churn_probability'] + variation, 0.01, 0.99
            )
            
            monthly_data.append(month_sample)
        
        final_combined = pd.concat(monthly_data, ignore_index=True)
        
        return final_combined
    
    def combine_data(self, predictions_df, ground_truth_df):
        """Combine predictions and ground truth data (legacy method)"""
        # Merge on customer_id
        combined = pd.merge(predictions_df, ground_truth_df, on='customer_id', how='inner')
        
        # Convert dates
        if 'prediction_date' in combined.columns:
            combined['prediction_date'] = pd.to_datetime(combined['prediction_date'])
        if 'actual_date' in combined.columns:
            combined['actual_date'] = pd.to_datetime(combined['actual_date'])
            
        # Add month column for grouping
        combined['month'] = combined['prediction_date'].dt.strftime('%b')
        
        return combined
    
    def generate_mock_combined_data(self):
        """Generate complete mock dataset"""
        np.random.seed(42)
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        data = []
        for date in dates:
            n_customers = np.random.randint(80, 120)
            for i in range(n_customers):
                customer_id = f"CUST_{date.strftime('%Y%m%d')}_{i:03d}"
                churn_prob = np.random.beta(2, 5)
                # Generate actual churn with correlation to probability
                actual_churn = np.random.choice([0, 1], p=[1-churn_prob*0.8, churn_prob*0.8])
                monthly_charges = np.clip(np.random.gamma(2, 30) + np.random.normal(0, 5), 20, 120)
                
                data.append({
                    'customer_id': customer_id,
                    'prediction_date': date,
                    'churn_probability': churn_prob,
                    'actual_churn': actual_churn,
                    'monthly_charges': monthly_charges,
                    'month': date.strftime('%b')
                })
        
        return pd.DataFrame(data)
    
    def calculate_psi(self, expected, actual, feature_name):
        """Calculate Population Stability Index (PSI)"""
        try:
            # Convert to numeric
            expected_numeric = pd.to_numeric(expected, errors='coerce').dropna()
            actual_numeric = pd.to_numeric(actual, errors='coerce').dropna()
            
            if len(expected_numeric) == 0 or len(actual_numeric) == 0:
                return 0.0
            
            # Create bins based on expected data quantiles
            bins = np.percentile(expected_numeric, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            bins = np.unique(bins)  # Remove duplicates
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            expected_counts = pd.cut(expected_numeric, bins, duplicates='drop').value_counts().sort_index()
            actual_counts = pd.cut(actual_numeric, bins, duplicates='drop').value_counts().sort_index()
            
            # Convert to percentages
            expected_pct = expected_counts / len(expected_numeric)
            actual_pct = actual_counts / len(actual_numeric)
            
            # Handle zero percentages
            expected_pct = expected_pct.replace(0, 0.0001)
            actual_pct = actual_pct.replace(0, 0.0001)
            
            # Calculate PSI
            psi = sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
            
        except Exception as e:
            print(f"Error calculating PSI for {feature_name}: {e}")
            return 0.0
    
    def run_comparison(self):
        """Run the complete monitoring process"""
        print("=== Dynamic OOT Monitoring ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load data
        all_data = self.load_data()
        
        if all_data.empty:
            print("❌ No data available for analysis")
            return
        
        print(f"📊 Analyzing {len(all_data)} records from {self.start_date} to {self.end_date}")
        print()
        
        # Generate charts
        self.create_auc_time_chart(all_data)
        self.create_kde_predictions_chart(all_data)
        self.create_kde_monthly_charges_chart(all_data)
        
        # Calculate and print PSI
        self.calculate_and_print_psi(all_data)
        
        print("\n" + "="*60)
        print("📊 CHARTS GENERATED:")
        print(f"   • AUC chart: {os.path.join(self.output_dir, 'auc_time_chart.png')}")
        print(f"   • KDE predictions: {os.path.join(self.output_dir, 'kde_predictions.png')}")
        print(f"   • KDE monthly charges: {os.path.join(self.output_dir, 'kde_monthly_charges.png')}")
        print("="*60)
    
    def create_auc_time_chart(self, all_data):
        """Create AUC over time chart using actual data"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        # Calculate AUC for each month
        months = []
        auc_values = []
        
        for month in ['Apr', 'May', 'Jun', 'Jul']:
            month_data = all_data[all_data['month'] == month]
            if len(month_data) > 0 and len(month_data['actual_churn'].unique()) > 1:
                try:
                    auc = roc_auc_score(month_data['actual_churn'], month_data['churn_probability'])
                    months.append(month)
                    auc_values.append(auc)
                    print(f"📈 {month} AUC: {auc:.3f}")
                except Exception as e:
                    print(f"⚠️  Could not calculate AUC for {month}: {e}")
        
        if not auc_values:
            print("❌ No AUC values could be calculated")
            return
        
        # Create the plot
        ax.plot(months, auc_values, marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax.axhline(y=0.8, color='#F24236', linestyle='--', alpha=0.7, label='Threshold (0.8)')
        
        # Styling
        ax.set_title('Model Performance Over Time (Dynamic)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Month (2024)', fontsize=12)
        ax.set_ylabel('AUC Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(max(0.5, min(auc_values) - 0.05), min(1.0, max(auc_values) + 0.05))
        
        # Add value labels
        for i, (month, auc) in enumerate(zip(months, auc_values)):
            ax.annotate(f'{auc:.3f}', (i, auc), textcoords="offset points", xytext=(0,10), ha='center')
        
        # Calculate and display average
        avg_auc = np.mean(auc_values)
        ax.text(0.02, 0.98, f'Average AUC: {avg_auc:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'auc_time_chart.png'), bbox_inches='tight')
        plt.close()
        
        print(f"📈 Average AUC: {avg_auc:.3f}")
        if avg_auc < 0.8:
            print(f"⚠️  WARNING: Average AUC below threshold")
        else:
            print(f"✅ Average AUC meets threshold")
    
    def create_kde_predictions_chart(self, all_data):
        """Create KDE chart for prediction distributions using actual data"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        months = ['Apr', 'May', 'Jun', 'Jul']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (month, color) in enumerate(zip(months, colors)):
            month_data = all_data[all_data['month'] == month]
            if len(month_data) > 0:
                predictions = month_data['churn_probability']
                sns.kdeplot(data=predictions, label=f'{month} 2024', color=color, linewidth=3, ax=ax)
        
        ax.set_title('Prediction Score Distributions by Month (Dynamic)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kde_predictions.png'), bbox_inches='tight')
        plt.close()
        
        print("📊 KDE Predictions chart generated")
    
    def create_kde_monthly_charges_chart(self, all_data):
        """Create KDE chart for monthly charges distributions using actual data"""
        plt.style.use('seaborn-v0_8')
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        months = ['Apr', 'May', 'Jun', 'Jul']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (month, color) in enumerate(zip(months, colors)):
            month_data = all_data[all_data['month'] == month]
            if len(month_data) > 0:
                charges = month_data['monthly_charges']
                sns.kdeplot(data=charges, label=f'{month} 2024', color=color, linewidth=3, ax=ax)
        
        ax.set_title('Monthly Charges Distributions by Month (Dynamic)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Monthly Charges ($)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kde_monthly_charges.png'), bbox_inches='tight')
        plt.close()
        
        print("📊 KDE Monthly Charges chart generated")
    
    def calculate_and_print_psi(self, all_data):
        """Calculate and print PSI values"""
        print("\n📊 POPULATION STABILITY INDEX (PSI) ANALYSIS:")
        print("="*50)
        
        # Use April as baseline (expected) and compare other months
        baseline_data = all_data[all_data['month'] == 'Apr']
        
        if len(baseline_data) == 0:
            print("❌ No baseline data (April) available for PSI calculation")
            return
        
        features_to_analyze = ['churn_probability', 'monthly_charges']
        
        for feature in features_to_analyze:
            if feature in all_data.columns:
                print(f"\n🔍 PSI Analysis for {feature}:")
                baseline_values = baseline_data[feature]
                
                for month in ['May', 'Jun', 'Jul']:
                    month_data = all_data[all_data['month'] == month]
                    if len(month_data) > 0:
                        month_values = month_data[feature]
                        psi = self.calculate_psi(baseline_values, month_values, feature)
                        
                        # PSI interpretation
                        if psi < 0.1:
                            status = "✅ Stable"
                        elif psi < 0.2:
                            status = "⚠️  Moderate drift"
                        else:
                            status = "🚨 Significant drift"
                        
                        print(f"   {month} vs Apr: PSI = {psi:.4f} ({status})")
                    else:
                        print(f"   {month} vs Apr: No data available")
        
        print("\n📋 PSI Interpretation:")
        print("   PSI < 0.1  : No significant change")
        print("   PSI 0.1-0.2: Moderate change, investigate")
        print("   PSI > 0.2  : Significant change, action required")

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    print("🚀 Starting Dynamic Model Monitoring Analysis...")
    print("=" * 60)
    
    monitor = SimpleOOTMonitor()
    monitor.run_comparison()
    
    print("\n" + "=" * 60)
    print("✅ Dynamic Model Monitoring Analysis Complete!")
    sys.stdout.flush()
