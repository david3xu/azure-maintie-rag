"""
GNN Model Quality Assessment Service
Enterprise model evaluation against raw data quality
"""
import torch
import numpy as np
from typing import Dict, Any, List
from torch_geometric.data import DataLoader
import logging
from config.settings import azure_settings

logger = logging.getLogger(__name__)

class GNNModelQualityAssessor:
    """Enterprise GNN model quality assessment service"""
    def assess_model_quality(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> Dict[str, Any]:
        quality_metrics = {}
        performance_metrics = self._evaluate_model_performance(model, data_loader)
        quality_metrics.update(performance_metrics)
        structure_metrics = self._evaluate_graph_understanding(model, data_loader, domain)
        quality_metrics.update(structure_metrics)
        domain_metrics = self._evaluate_domain_quality(model, data_loader, domain)
        quality_metrics.update(domain_metrics)
        overall_score = self._calculate_overall_quality_score(quality_metrics)
        quality_metrics["overall_quality_score"] = overall_score
        recommendations = self._generate_quality_recommendations(quality_metrics)
        quality_metrics["quality_recommendations"] = recommendations
        return quality_metrics

    def _evaluate_model_performance(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in data_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                pred = out.argmax(dim=1)
                correct += pred.eq(batch.y).sum().item()
                total += batch.y.size(0)
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(batch.y.cpu().numpy())
        accuracy = correct / total if total > 0 else 0.0
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted', zero_division=0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total_samples": total
        }

    def _evaluate_graph_understanding(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> Dict[str, float]:
        embeddings_quality = self._assess_embedding_quality(model, data_loader)
        connectivity_score = self._assess_connectivity_understanding(model, data_loader, domain)
        return {
            "embedding_quality": embeddings_quality,
            "connectivity_understanding": connectivity_score,
            "graph_structure_score": (embeddings_quality + connectivity_score) / 2
        }

    def _evaluate_domain_quality(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> Dict[str, float]:
        entity_recognition_score = self._assess_entity_recognition(model, data_loader, domain)
        relationship_score = self._assess_relationship_understanding(model, data_loader, domain)
        connectivity_score = self._assess_connectivity_understanding(model, data_loader, domain)
        return {
            f"{domain}_entity_recognition": entity_recognition_score,
            f"{domain}_relationship_understanding": relationship_score,
            f"{domain}_connectivity_understanding": connectivity_score,
            f"{domain}_domain_score": (entity_recognition_score + relationship_score + connectivity_score) / 3
        }

    def _assess_embedding_quality(self, model: torch.nn.Module, data_loader: DataLoader) -> float:
        embeddings = []
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                x = batch.x
                for layer in model.convs[:-1]:
                    x = layer(x, batch.edge_index)
                embeddings.append(x.cpu().numpy())
        if not embeddings:
            return 0.0
        all_embeddings = np.vstack(embeddings)
        embedding_std = np.std(all_embeddings, axis=0).mean()
        embedding_diversity = min(1.0, embedding_std / 0.5)
        return embedding_diversity

    def _assess_connectivity_understanding(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> float:
        """Assess connectivity understanding using actual graph topology from Cosmos DB"""
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            cosmos_client = AzureCosmosGremlinClient()
            connectivity_query = f"""
                g.V().has('domain', '{domain}')
                    .project('degree_centrality', 'clustering_coefficient')
                    .by(__.bothE().count())
                    .by(__.local(__.bothE().otherV().aggregate('neighbors').bothE().otherV().where(__.within('neighbors')).count().math('_/2')))
            """
            connectivity_stats = cosmos_client._execute_gremlin_query_safe(connectivity_query)
            if not connectivity_stats:
                return 0.0
            avg_degree = sum(stat.get('degree_centrality', 0) for stat in connectivity_stats) / len(connectivity_stats)
            avg_clustering = sum(stat.get('clustering_coefficient', 0) for stat in connectivity_stats) / len(connectivity_stats)
            connectivity_score = min(1.0, (avg_degree * 0.6 + avg_clustering * 0.4) / 10.0)
            return connectivity_score
        except Exception as e:
            logger.error(f"Connectivity assessment failed: {e}")
            return 0.0

    def _assess_entity_recognition(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> float:
        """Assess entity recognition using confidence scores from Cosmos DB"""
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            cosmos_client = AzureCosmosGremlinClient()
            entity_confidence_query = f"""
                g.V().has('domain', '{domain}')
                    .has('confidence')
                    .values('confidence')
            """
            confidence_scores = cosmos_client._execute_gremlin_query_safe(entity_confidence_query)
            if not confidence_scores:
                return 0.0
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_variance = sum((score - avg_confidence) ** 2 for score in confidence_scores) / len(confidence_scores)
            entity_recognition_score = avg_confidence * (1.0 - min(0.5, confidence_variance))
            return min(1.0, entity_recognition_score)
        except Exception as e:
            logger.error(f"Entity recognition assessment failed: {e}")
            return 0.0

    def _assess_relationship_understanding(self, model: torch.nn.Module, data_loader: DataLoader, domain: str) -> float:
        """Assess relationship understanding using actual relationship patterns from Cosmos DB"""
        try:
            from core.azure_cosmos.cosmos_gremlin_client import AzureCosmosGremlinClient
            cosmos_client = AzureCosmosGremlinClient()
            relationship_analysis_query = f"""
                g.E().has('domain', '{domain}')
                    .group().by('relation_type').by(__.count())
            """
            relation_type_counts = cosmos_client._execute_gremlin_query_safe(relationship_analysis_query)
            if not relation_type_counts:
                return 0.0
            total_relations = sum(relation_type_counts.values())
            unique_types = len(relation_type_counts)
            type_diversity = min(1.0, unique_types / 10.0)
            if total_relations > 0:
                max_type_ratio = max(relation_type_counts.values()) / total_relations
                distribution_score = 1.0 - max_type_ratio
            else:
                distribution_score = 0.0
            relationship_understanding_score = (type_diversity * 0.6 + distribution_score * 0.4)
            return min(1.0, relationship_understanding_score)
        except Exception as e:
            logger.error(f"Relationship understanding assessment failed: {e}")
            return 0.0

    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        performance_weight = 0.4
        structure_weight = 0.3
        domain_weight = 0.3
        performance_score = metrics.get("f1_score", 0.0)
        structure_score = metrics.get("graph_structure_score", 0.0)
        domain_score = 0.0
        for key, value in metrics.items():
            if key.endswith("_domain_score"):
                domain_score = value
                break
        overall_score = (
            performance_weight * performance_score +
            structure_weight * structure_score +
            domain_weight * domain_score
        )
        return overall_score

    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        recommendations = []
        quality_threshold = getattr(azure_settings, 'gnn_quality_threshold', 0.6)
        if metrics.get("accuracy", 0) < 0.7:
            recommendations.append("Consider increasing model complexity or training epochs")
        if metrics.get("embedding_quality", 0) < 0.5:
            recommendations.append("Embedding collapse detected - adjust learning rate or add regularization")
        if metrics.get("connectivity_understanding", 0) < quality_threshold:
            recommendations.append(f"Graph connectivity understanding below threshold ({quality_threshold}) - review graph structure quality")
        if metrics.get("overall_quality_score", 0) < quality_threshold:
            recommendations.append(f"Overall model quality below environment threshold ({quality_threshold}) - review data quality and model architecture")
        return recommendations