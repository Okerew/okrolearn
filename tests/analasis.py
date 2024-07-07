from okrolearn.src.okrolearn.okrolearn import Tensor, np
from okrolearn.src.okrolearn.okrolanalysis import DataAnalyzer
data = Tensor(np.random.randn(100, 5))
target = np.random.rand(100)
analyzer = DataAnalyzer(data)
analyzer.plot_correlation_heatmap()
analyzer.plot_pca()
analyzer.plot_kmeans()
analyzer.histogram()
analyzer.scatter_matrix()
analyzer.plot_outliers()
analyzer.plot_feature_importance(target)
analyzer.analyze_patterns(feature_index=0)
analyzer.analyze_patterns(feature_index=1)
analyzer.analyze_time_series(feature_index=0)
analyzer.detect_seasonality(feature_index=0)
analyzer.trend_analysis(feature_index=0)
