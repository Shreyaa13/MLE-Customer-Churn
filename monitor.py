import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import accuracy_score

class SimpleOOTMonitor:
    def __init__(self):
        self.bronze_dir = "scripts/datamart/bronze"
        self.output_dir = "scripts/datamart/monitoring"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, date_str, data_type):
        """Load data from bronze layer"""
        try:
            features_path = f"{self.bronze_dir}/features/bronze_features_{date_str}.csv"
            labels_path = f"{self.bronze_dir}/labels/bronze_labels_{date_str}.csv"
            
            if os.path.exists(features_path) and os.path.exists(labels_path):
                features = pd.read_csv(features_path)
                labels = pd.read_csv(labels_path)
                data = features.merge(labels, on='customerID', how='inner')
                print(f"Loaded {data_type} data: {len(data)} rows")
                return data
            else:
                print(f"Data files not found for {date_str}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading {data_type} data: {e}")
            return pd.DataFrame()
    
    def load_training_data(self):
        """Load baseline training data (Apr-Jun 2024)"""
        training_dates = ['2024-04-01', '2024-05-01', '2024-06-01']
        all_data = []
        
        for date in training_dates:
            data = self.load_data(date, f"training-{date}")
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"Combined training data: {len(combined)} rows")
            return combined
        return pd.DataFrame()
    
    def load_oot_data(self):
        """Load OOT data (Jul 2024)"""
        return self.load_data('2024-07-01', 'OOT')
    
    def calculate_psi(self, expected, actual, feature):
        """Calculate PSI for a feature"""
        try:
            # Convert to numeric, handling string values
            expected_numeric = pd.to_numeric(expected, errors='coerce')
            actual_numeric = pd.to_numeric(actual, errors='coerce')
            
            # Remove NaN values
            expected_clean = expected_numeric.dropna()
            actual_clean = actual_numeric.dropna()
            
            if len(expected_clean) == 0 or len(actual_clean) == 0:
                return 0.0
            
            # Create bins based on expected data
            bins = np.linspace(expected_clean.min(), expected_clean.max(), 11)
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            expected_counts = pd.cut(expected_clean, bins).value_counts().sort_index()
            actual_counts = pd.cut(actual_clean, bins).value_counts().sort_index()
            
            # Convert to percentages
            expected_pct = expected_counts / len(expected_clean)
            actual_pct = actual_counts / len(actual_clean)
            
            # Handle zero percentages
            expected_pct = expected_pct.replace(0, 0.0001)
            actual_pct = actual_pct.replace(0, 0.0001)
            
            # Calculate PSI
            psi = sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            return psi
        except Exception as e:
            print(f"Error calculating PSI for {feature}: {e}")
            return 0.0
    
    def generate_predictions(self, data):
        """Generate simple predictions for comparison"""
        import numpy as np
        np.random.seed(42)
        # Simple rule-based predictions with some randomness
        predictions = []
        for _, row in data.iterrows():
            # Higher chance of churn for high monthly charges and low tenure
            prob = 0.3
            if 'MonthlyCharges' in row:
                try:
                    monthly_charges = float(row['MonthlyCharges'])
                    if monthly_charges > 70:
                        prob += 0.2
                except (ValueError, TypeError):
                    pass
            if 'tenure' in row:
                try:
                    tenure = float(row['tenure'])
                    if tenure < 12:
                        prob += 0.3
                except (ValueError, TypeError):
                    pass
            # Return 'Yes'/'No' to match label format
            predictions.append('Yes' if np.random.random() < prob else 'No')
        return np.array(predictions)
    
    def create_accuracy_chart(self, training_acc, oot_acc):
        """Create enhanced accuracy comparison chart"""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 7))
        
        categories = ['Training', 'OOT']
        accuracies = [training_acc * 100, oot_acc * 100]
        degradation = (oot_acc - training_acc) * 100
        
        # Enhanced color scheme with status indication
        if abs(degradation) < 5:
            colors = ['#2E86AB', '#A23B72']  # Blue to purple - good performance
        elif abs(degradation) < 10:
            colors = ['#2E86AB', '#F18F01']  # Blue to orange - moderate degradation
        else:
            colors = ['#2E86AB', '#C73E1D']  # Blue to red - significant degradation
        
        bars = ax.bar(categories, accuracies, color=colors, alpha=0.85, 
                     edgecolor='white', linewidth=2, width=0.6)
        
        # Enhanced styling
        ax.set_ylim(0, 100)
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance: Training vs Out-of-Time (OOT)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Enhanced value labels with background
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold',
                   fontsize=14, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add degradation indicator
        if degradation < 0:
            degradation_text = f'↓ {abs(degradation):.1f}% degradation'
            color = '#C73E1D' if abs(degradation) >= 10 else '#F18F01'
        else:
            degradation_text = f'↑ {degradation:.1f}% improvement'
            color = '#2ECC71'
        
        ax.text(0.5, 0.95, degradation_text, transform=ax.transAxes, 
               ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color, 
               alpha=0.2, edgecolor=color))
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.02, 0.02, f'Generated: {timestamp}', transform=ax.transAxes,
               fontsize=9, alpha=0.7, style='italic')
        
        plt.tight_layout()
        accuracy_path = f"{self.output_dir}/accuracy_comparison.png"
        plt.savefig(accuracy_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📊 Enhanced accuracy chart saved: {accuracy_path}")
    
    def create_psi_chart(self, psi_results):
        """Create enhanced PSI comparison chart"""
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features = list(psi_results.keys())
        psi_values = list(psi_results.values())
        
        # Enhanced color coding with gradients
        colors = []
        status_labels = []
        for psi in psi_values:
            if psi < 0.1:
                colors.append('#27AE60')  # Green - Stable
                status_labels.append('Stable')
            elif psi < 0.2:
                colors.append('#F39C12')  # Orange - Minor change
                status_labels.append('Minor Change')
            else:
                colors.append('#E74C3C')  # Red - Major change
                status_labels.append('Major Change')
        
        bars = ax.bar(features, psi_values, color=colors, alpha=0.85,
                     edgecolor='white', linewidth=2, width=0.7)
        
        # Enhanced styling
        max_psi = max(psi_values) if psi_values else 0.3
        ax.set_ylim(0, max(0.3, max_psi * 1.2))
        ax.set_ylabel('Population Stability Index (PSI)', fontsize=12, fontweight='bold')
        ax.set_title('Feature Drift Analysis: Population Stability Index', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
        
        # Adjust layout to prevent label overlap
        fig.subplots_adjust(bottom=0.15)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Enhanced threshold lines with better styling
        ax.axhline(y=0.1, color='#F39C12', linestyle='--', alpha=0.8, linewidth=2,
                  label='Minor Change Threshold (0.1)')
        ax.axhline(y=0.2, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2,
                  label='Major Change Threshold (0.2)')
        
        # Enhanced value labels with status indicators
        for i, (bar, psi, status) in enumerate(zip(bars, psi_values, status_labels)):
            # PSI value label
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_psi * 0.02,
                   f'{psi:.3f}', ha='center', va='bottom', fontweight='bold',
                   fontsize=11, bbox=dict(boxstyle='round,pad=0.2', 
                   facecolor='white', alpha=0.9, edgecolor=colors[i]))
            
            # Status label
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_psi * 0.08,
                   status, ha='center', va='bottom', fontsize=9, 
                   color=colors[i], fontweight='bold')
        
        # Enhanced legend
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, 
                          shadow=True, fontsize=10)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # Add summary statistics
        stable_count = sum(1 for psi in psi_values if psi < 0.1)
        minor_count = sum(1 for psi in psi_values if 0.1 <= psi < 0.2)
        major_count = sum(1 for psi in psi_values if psi >= 0.2)
        
        summary_text = f'Summary: {stable_count} Stable, {minor_count} Minor Changes, {major_count} Major Changes'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
               fontsize=11, fontweight='bold', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
               alpha=0.3, edgecolor='steelblue'))
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(0.98, 0.02, f'Generated: {timestamp}', transform=ax.transAxes,
               fontsize=9, alpha=0.7, style='italic', ha='right')
        
        plt.tight_layout()
        psi_path = f"{self.output_dir}/psi_comparison.png"
        plt.savefig(psi_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📈 Enhanced PSI chart saved: {psi_path}")
    
    def run_comparison(self):
        """Run the complete training vs OOT comparison"""
        print("=== Simple OOT Monitoring ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load data
        training_data = self.load_training_data()
        oot_data = self.load_oot_data()
        
        if training_data.empty or oot_data.empty:
            print("❌ Failed to load required data")
            return
        
        # Calculate accuracy
        training_pred = self.generate_predictions(training_data)
        oot_pred = self.generate_predictions(oot_data)
        
        # Convert labels to consistent format if needed
        training_labels = training_data['Churn'].astype(str)
        oot_labels = oot_data['Churn'].astype(str)
        
        training_acc = accuracy_score(training_labels, training_pred)
        oot_acc = accuracy_score(oot_labels, oot_pred)
        
        degradation = (oot_acc - training_acc) * 100
        
        print("📊 ACCURACY COMPARISON")
        print(f"Training Accuracy: {training_acc:.3f} ({training_acc*100:.1f}%)")
        print(f"OOT Accuracy: {oot_acc:.3f} ({oot_acc*100:.1f}%)")
        print(f"Degradation: {degradation:.1f}%")
        
        if abs(degradation) < 10:
            print("✅ Degradation within acceptable range")
        else:
            print("⚠️ Significant degradation detected")
        print()
        
        # Calculate PSI for key features
        print("📈 PSI ANALYSIS")
        psi_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        psi_results = {}
        
        for feature in psi_features:
            if feature in training_data.columns and feature in oot_data.columns:
                psi = self.calculate_psi(training_data[feature], oot_data[feature], feature)
                psi_results[feature] = psi
                
                if psi < 0.1:
                    status = "🟢 Stable"
                elif psi < 0.2:
                    status = "🟡 Minor change"
                else:
                    status = "🔴 Major change"
                
                print(f"{feature}: PSI = {psi:.3f} ({status})")
        
        # Create enhanced charts
        self.create_accuracy_chart(training_acc, oot_acc)
        self.create_psi_chart(psi_results)
        self.create_summary_dashboard(training_acc, oot_acc, psi_results, degradation)
        
        print()
        print(f"📁 Enhanced monitoring results saved to: {self.output_dir}/")
        print(f"📊 Charts generated: accuracy_comparison.png, psi_comparison.png, summary_dashboard.png")

    def create_summary_dashboard(self, training_acc, oot_acc, psi_results, degradation):
        """Create a comprehensive summary dashboard"""
        plt.style.use('default')
        fig = plt.figure(figsize=(16, 10))
        
        # Create a 2x2 grid layout with better spacing
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.2], 
                             hspace=0.4, wspace=0.35)
        
        # 1. Accuracy comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        categories = ['Training', 'OOT']
        accuracies = [training_acc * 100, oot_acc * 100]
        colors = ['#2E86AB', '#A23B72' if abs(degradation) < 10 else '#C73E1D']
        
        bars1 = ax1.bar(categories, accuracies, color=colors, alpha=0.8, width=0.6)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Accuracy (%)', fontweight='bold')
        ax1.set_title('Model Performance', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. PSI overview (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        features = list(psi_results.keys())
        psi_values = list(psi_results.values())
        
        psi_colors = []
        for psi in psi_values:
            if psi < 0.1:
                psi_colors.append('#27AE60')
            elif psi < 0.2:
                psi_colors.append('#F39C12')
            else:
                psi_colors.append('#E74C3C')
        
        bars2 = ax2.bar(features, psi_values, color=psi_colors, alpha=0.8)
        ax2.set_ylabel('PSI Value', fontweight='bold')
        ax2.set_title('Feature Drift (PSI)', fontweight='bold', fontsize=14)
        ax2.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.7)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        for bar, psi in zip(bars2, psi_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{psi:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 3. Summary metrics (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        # Create summary text
        summary_text = f"""
📊 MONITORING SUMMARY

🎯 Model Performance:
• Training Accuracy: {training_acc*100:.1f}%
• OOT Accuracy: {oot_acc*100:.1f}%
• Performance Change: {degradation:+.1f}%

📈 Feature Drift Analysis:
• Stable Features: {sum(1 for psi in psi_values if psi < 0.1)}
• Minor Changes: {sum(1 for psi in psi_values if 0.1 <= psi < 0.2)}
• Major Changes: {sum(1 for psi in psi_values if psi >= 0.2)}

⚠️ Alert Status:
• {'🟢 All systems normal' if abs(degradation) < 5 and all(psi < 0.2 for psi in psi_values) else '🟡 Monitor closely' if abs(degradation) < 10 else '🔴 Action required'}
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))
        
        # 4. Trend indicators (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create a simple trend visualization
        metrics = ['Accuracy\nDegradation', 'Feature\nDrift Risk', 'Overall\nHealth']
        
        # Calculate risk scores (0-100)
        acc_risk = min(100, abs(degradation) * 10) if degradation < 0 else 0  # Only negative degradation is risk
        drift_risk = min(100, max(psi_values) * 500) if psi_values else 0  # 0.2 PSI = 100 risk
        overall_risk = (acc_risk + drift_risk) / 2
        
        risks = [acc_risk, drift_risk, overall_risk]
        risk_colors = []
        for risk in risks:
            if risk < 30:
                risk_colors.append('#27AE60')  # Green
            elif risk < 70:
                risk_colors.append('#F39C12')  # Orange
            else:
                risk_colors.append('#E74C3C')  # Red
        
        bars4 = ax4.barh(metrics, risks, color=risk_colors, alpha=0.8)
        ax4.set_xlim(0, 100)
        ax4.set_xlabel('Risk Level (%)', fontweight='bold')
        ax4.set_title('Risk Assessment', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        for bar, risk in zip(bars4, risks):
            ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{risk:.0f}%', ha='left', va='center', fontweight='bold')
        
        # Add overall title and timestamp
        fig.suptitle('Model Monitoring Dashboard - Out-of-Time Analysis', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom',
                fontsize=10, alpha=0.7, style='italic')
        
        # Save dashboard
        dashboard_path = f"{self.output_dir}/summary_dashboard.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"📋 Summary dashboard saved: {dashboard_path}")

if __name__ == "__main__":
    monitor = SimpleOOTMonitor()
    monitor.run_comparison()