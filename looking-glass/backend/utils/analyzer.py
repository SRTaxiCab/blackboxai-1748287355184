import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from transformers import pipeline
import logging
import torch

class DataAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_models()

    def setup_models(self):
        """Initialize NLP models and analysis pipelines"""
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )

            # Initialize zero-shot classification pipeline for topic analysis
            self.topic_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )

            # Initialize text generation pipeline for narrative analysis
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )

        except Exception as e:
            self.logger.error(f"Error setting up models: {str(e)}")
            raise

    def analyze_data(
        self,
        raw_data: Dict[str, Any],
        analysis_mode: str = "standard"
    ) -> Dict[str, Any]:
        """
        Main analysis pipeline for processing collected data
        """
        try:
            # Convert raw data to pandas DataFrame for analysis
            processed_data = self._preprocess_data(raw_data)
            
            # Perform different types of analysis
            analysis_results = {
                "sentiment_analysis": self._analyze_sentiment(processed_data),
                "trend_analysis": self._analyze_trends(processed_data),
                "anomaly_detection": self._detect_anomalies(processed_data),
                "narrative_analysis": self._analyze_narratives(processed_data),
                "temporal_patterns": self._analyze_temporal_patterns(processed_data)
            }

            # Add mode-specific analysis
            if analysis_mode != "standard":
                analysis_results.update(
                    self._perform_special_analysis(processed_data, analysis_mode)
                )

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in data analysis: {str(e)}")
            raise

    def _preprocess_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess and combine data from different sources into a unified format
        """
        try:
            # Process news data
            news_df = pd.DataFrame(raw_data.get("news_data", []))
            if not news_df.empty:
                news_df['source_type'] = 'news'

            # Process social media data
            social_data = raw_data.get("social_media_data", {})
            social_dfs = []
            
            # Process Twitter data
            if "twitter" in social_data:
                twitter_df = pd.DataFrame(social_data["twitter"])
                if not twitter_df.empty:
                    twitter_df['source_type'] = 'twitter'
                    social_dfs.append(twitter_df)

            # Process Reddit data
            if "reddit" in social_data:
                reddit_df = pd.DataFrame(social_data["reddit"])
                if not reddit_df.empty:
                    reddit_df['source_type'] = 'reddit'
                    social_dfs.append(reddit_df)

            # Combine all data sources
            all_dfs = [news_df] + social_dfs
            combined_df = pd.concat(all_dfs, ignore_index=True)

            # Clean and standardize timestamps
            combined_df['timestamp'] = pd.to_datetime(combined_df['published_at'])
            combined_df = combined_df.sort_values('timestamp')

            return combined_df

        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _analyze_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform sentiment analysis on text content
        """
        try:
            texts = data['content'].fillna('').tolist()
            sentiments = []
            
            # Process texts in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_sentiments = self.sentiment_analyzer(batch)
                sentiments.extend(batch_sentiments)

            # Aggregate sentiment results
            sentiment_counts = {
                "positive": len([s for s in sentiments if s['label'] == 'POSITIVE']),
                "negative": len([s for s in sentiments if s['label'] == 'NEGATIVE'])
            }

            # Calculate temporal sentiment trends
            temporal_sentiment = data.groupby(pd.Grouper(key='timestamp', freq='D')).agg({
                'content': lambda x: self._get_daily_sentiment(x)
            }).to_dict()

            return {
                "overall_sentiment": sentiment_counts,
                "temporal_sentiment": temporal_sentiment
            }

        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {}

    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify and analyze emerging trends
        """
        try:
            # Extract key topics and their frequencies
            topics = self._extract_topics(data['content'].fillna('').tolist())
            
            # Analyze topic evolution over time
            temporal_trends = self._analyze_topic_evolution(data)
            
            # Identify emerging trends
            emerging_trends = self._identify_emerging_trends(temporal_trends)

            return {
                "current_topics": topics,
                "temporal_trends": temporal_trends,
                "emerging_trends": emerging_trends
            }

        except Exception as e:
            self.logger.error(f"Error in trend analysis: {str(e)}")
            return {}

    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect anomalies and unusual patterns in the data
        """
        try:
            # Prepare features for anomaly detection
            features = self._prepare_anomaly_features(data)
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features)

            # Apply DBSCAN for anomaly detection
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(normalized_features)

            # Identify anomalies (points labeled as -1 by DBSCAN)
            anomaly_indices = np.where(labels == -1)[0]
            anomalies = data.iloc[anomaly_indices]

            return {
                "anomaly_count": len(anomaly_indices),
                "anomaly_details": anomalies.to_dict('records'),
                "anomaly_timestamps": anomalies['timestamp'].tolist()
            }

        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {}

    def _analyze_narratives(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze narrative patterns and story development
        """
        try:
            # Group content by source and time
            narrative_groups = data.groupby(['source_type', pd.Grouper(key='timestamp', freq='D')])
            
            # Extract main narratives
            narratives = self._extract_main_narratives(narrative_groups)
            
            # Analyze narrative evolution
            narrative_evolution = self._analyze_narrative_evolution(narratives)

            return {
                "main_narratives": narratives,
                "narrative_evolution": narrative_evolution
            }

        except Exception as e:
            self.logger.error(f"Error in narrative analysis: {str(e)}")
            return {}

    def _analyze_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns and cyclical behaviors
        """
        try:
            # Resample data to daily frequency
            daily_counts = data.resample('D', on='timestamp').size()
            
            # Detect seasonal patterns
            seasonal_patterns = self._detect_seasonal_patterns(daily_counts)
            
            # Identify trend components
            trend_components = self._decompose_time_series(daily_counts)

            return {
                "daily_activity": daily_counts.to_dict(),
                "seasonal_patterns": seasonal_patterns,
                "trend_components": trend_components
            }

        except Exception as e:
            self.logger.error(f"Error in temporal pattern analysis: {str(e)}")
            return {}

    def _perform_special_analysis(
        self,
        data: pd.DataFrame,
        analysis_mode: str
    ) -> Dict[str, Any]:
        """
        Perform mode-specific analysis based on the selected analysis mode
        """
        try:
            if analysis_mode == "omniview":
                return self._perform_omniview_analysis(data)
            elif analysis_mode == "singularity_watch":
                return self._perform_singularity_analysis(data)
            elif analysis_mode == "prophetic_overlay":
                return self._perform_prophetic_analysis(data)
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error in special analysis mode: {str(e)}")
            return {}

    def _get_daily_sentiment(self, texts: List[str]) -> float:
        """
        Calculate average sentiment score for a group of texts
        """
        try:
            sentiments = self.sentiment_analyzer(texts.tolist())
            scores = [1 if s['label'] == 'POSITIVE' else 0 for s in sentiments]
            return np.mean(scores) if scores else 0.5

        except Exception as e:
            self.logger.error(f"Error in daily sentiment calculation: {str(e)}")
            return 0.5

    def _extract_topics(self, texts: List[str]) -> Dict[str, float]:
        """
        Extract main topics from texts using zero-shot classification
        """
        try:
            candidate_topics = [
                "politics", "technology", "economy", "social_issues",
                "environment", "health", "science", "culture"
            ]
            
            # Classify texts into topics
            classifications = self.topic_classifier(
                texts[:100],  # Limit to avoid memory issues
                candidate_topics,
                multi_label=True
            )

            # Aggregate topic scores
            topic_scores = {topic: 0.0 for topic in candidate_topics}
            for result in classifications:
                for topic, score in zip(result['labels'], result['scores']):
                    topic_scores[topic] += score

            # Normalize scores
            total = sum(topic_scores.values())
            if total > 0:
                topic_scores = {k: v/total for k, v in topic_scores.items()}

            return topic_scores

        except Exception as e:
            self.logger.error(f"Error in topic extraction: {str(e)}")
            return {}

    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection
        """
        try:
            # Extract numerical features
            features = []
            
            # Add temporal features
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            
            # Add content-based features
            data['content_length'] = data['content'].fillna('').str.len()
            
            # Combine features
            feature_matrix = np.column_stack([
                data['hour'],
                data['day_of_week'],
                data['content_length']
            ])

            return feature_matrix

        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            return np.array([])

    def _extract_main_narratives(self, narrative_groups) -> List[Dict[str, Any]]:
        """
        Extract main narratives from grouped content
        """
        narratives = []
        try:
            for name, group in narrative_groups:
                if len(group) > 0:
                    # Combine content for narrative analysis
                    combined_content = " ".join(group['content'].fillna(''))
                    
                    # Generate narrative summary
                    summary = self.text_generator(
                        combined_content[:1000],
                        max_length=100,
                        num_return_sequences=1
                    )[0]['generated_text']

                    narratives.append({
                        'source': name[0],
                        'date': name[1],
                        'summary': summary,
                        'content_count': len(group)
                    })

        except Exception as e:
            self.logger.error(f"Error in narrative extraction: {str(e)}")

        return narratives

    def _detect_seasonal_patterns(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time series data
        """
        try:
            # Calculate daily, weekly, and monthly patterns
            patterns = {
                'daily': time_series.groupby(time_series.index.hour).mean().to_dict(),
                'weekly': time_series.groupby(time_series.index.dayofweek).mean().to_dict(),
                'monthly': time_series.groupby(time_series.index.month).mean().to_dict()
            }
            return patterns

        except Exception as e:
            self.logger.error(f"Error in seasonal pattern detection: {str(e)}")
            return {}

    def _decompose_time_series(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components
        """
        try:
            # Perform time series decomposition
            # This is a simplified version - could be expanded with more sophisticated methods
            rolling_mean = time_series.rolling(window=7).mean()
            trend = rolling_mean.fillna(method='bfill').fillna(method='ffill')
            residual = time_series - trend

            return {
                'trend': trend.to_dict(),
                'residual': residual.to_dict()
            }

        except Exception as e:
            self.logger.error(f"Error in time series decomposition: {str(e)}")
            return {}
