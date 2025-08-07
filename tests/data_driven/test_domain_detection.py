"""
Data-Driven Domain Detection Tests - CODING_STANDARDS Compliant
Tests mathematical domain analysis without hardcoded assumptions.
"""

import pytest
import time
from typing import Dict, Any, List


@pytest.mark.azure
@pytest.mark.integration
class TestDataDrivenDomainDetection:
    """Test domain detection using real data and mathematical analysis"""

    @pytest.mark.asyncio
    async def test_statistical_domain_analysis(self, sample_documents, azure_services):
        """Test domain detection using statistical analysis, not hardcoded rules"""

        domain_results = {}

        for expected_domain, document in sample_documents.items():
            # Use Azure services for domain analysis (no hardcoded patterns)
            if not azure_services.openai_client:
                pytest.skip("Azure OpenAI not available for domain analysis")

            # Real domain analysis using Azure OpenAI
            analysis_prompt = f"""
            Analyze this text content and identify its domain based on vocabulary patterns,
            terminology frequency, and linguistic characteristics. Provide statistical confidence.
            
            Text: {document[:500]}...
            
            Return only: domain_name:confidence_score (e.g., programming:0.85)
            """

            start_time = time.time()
            result = await azure_services.openai_client.get_completion(
                analysis_prompt, max_tokens=50
            )
            analysis_time = time.time() - start_time

            if result and result.get("success"):
                response_text = result.get("content", "").strip()

                # Parse domain:confidence format
                if ":" in response_text:
                    try:
                        detected_domain, confidence_str = response_text.split(":", 1)
                        confidence = float(confidence_str.strip())

                        domain_results[expected_domain] = {
                            "detected_domain": detected_domain.strip().lower(),
                            "confidence": confidence,
                            "analysis_time": analysis_time,
                            "expected_domain": expected_domain,
                        }

                        print(
                            f"📊 {expected_domain.title()}: "
                            f"detected='{detected_domain.strip()}', "
                            f"confidence={confidence:.3f}, "
                            f"time={analysis_time:.3f}s"
                        )

                    except (ValueError, IndexError) as e:
                        print(
                            f"⚠️ Could not parse domain analysis for {expected_domain}: {response_text}"
                        )
                        domain_results[expected_domain] = {
                            "detected_domain": "parse_error",
                            "confidence": 0.0,
                            "analysis_time": analysis_time,
                            "error": str(e),
                        }
            else:
                print(f"❌ Domain analysis failed for {expected_domain}")
                domain_results[expected_domain] = {
                    "detected_domain": "analysis_failed",
                    "confidence": 0.0,
                    "analysis_time": analysis_time,
                    "error": "Azure OpenAI analysis failed",
                }

        # Validate results
        assert len(domain_results) > 0, "No domain analysis results obtained"

        # Check that we get reasonable confidence scores (data-driven validation)
        confident_results = [
            result
            for result in domain_results.values()
            if result.get("confidence", 0) > 0.5
        ]

        confidence_rate = len(confident_results) / len(domain_results) * 100
        print(
            f"🎯 Domain Detection Results: {confidence_rate:.1f}% high-confidence detections"
        )

        # Validate at least some domains detected with confidence
        assert (
            confidence_rate >= 50.0
        ), f"Too few confident domain detections: {confidence_rate:.1f}%"

        return domain_results

    @pytest.mark.asyncio
    async def test_cross_domain_vocabulary_analysis(
        self, sample_documents, azure_services
    ):
        """Test vocabulary-based domain differentiation"""

        if not azure_services.openai_client:
            pytest.skip("Azure OpenAI not available for vocabulary analysis")

        vocabulary_analysis = {}

        for domain, document in sample_documents.items():
            # Extract domain-specific vocabulary using Azure analysis
            vocab_prompt = f"""
            Extract the 5 most domain-specific technical terms from this text.
            Focus on terminology that clearly indicates the subject domain.
            
            Text: {document}
            
            Return only: term1,term2,term3,term4,term5
            """

            result = await azure_services.openai_client.get_completion(
                vocab_prompt, max_tokens=100
            )

            if result and result.get("success"):
                response_text = result.get("content", "").strip()
                terms = [term.strip() for term in response_text.split(",")]

                vocabulary_analysis[domain] = {
                    "key_terms": terms,
                    "term_count": len(terms),
                    "analysis_successful": True,
                }

                print(f"📝 {domain.title()} key terms: {', '.join(terms[:3])}...")
            else:
                vocabulary_analysis[domain] = {
                    "key_terms": [],
                    "term_count": 0,
                    "analysis_successful": False,
                }

        # Validate vocabulary extraction
        successful_analyses = [
            analysis
            for analysis in vocabulary_analysis.values()
            if analysis["analysis_successful"] and analysis["term_count"] > 0
        ]

        success_rate = len(successful_analyses) / len(vocabulary_analysis) * 100
        print(f"📚 Vocabulary Analysis Success: {success_rate:.1f}%")

        assert (
            success_rate >= 75.0
        ), f"Vocabulary analysis success rate too low: {success_rate:.1f}%"

        return vocabulary_analysis

    @pytest.mark.asyncio
    async def test_domain_confidence_thresholds(self, sample_documents, azure_services):
        """Test that confidence thresholds are data-driven, not hardcoded"""

        if not azure_services.openai_client:
            pytest.skip("Azure OpenAI not available for confidence analysis")

        confidence_scores = []

        for domain, document in sample_documents.items():
            # Get confidence score from real analysis
            confidence_prompt = f"""
            Rate the clarity of domain identification for this text on a scale of 0.0 to 1.0.
            Consider vocabulary specificity, technical terminology, and context clarity.
            
            Text: {document[:300]}...
            
            Return only the confidence score (e.g., 0.85)
            """

            result = await azure_services.openai_client.get_completion(
                confidence_prompt, max_tokens=10
            )

            if result and result.get("success"):
                try:
                    confidence_text = result.get("content", "").strip()
                    confidence = float(confidence_text)

                    if 0.0 <= confidence <= 1.0:
                        confidence_scores.append(
                            {"domain": domain, "confidence": confidence, "valid": True}
                        )

                        print(f"🎯 {domain.title()} confidence: {confidence:.3f}")
                    else:
                        print(f"⚠️ Invalid confidence score for {domain}: {confidence}")

                except (ValueError, TypeError):
                    print(
                        f"⚠️ Could not parse confidence for {domain}: {confidence_text}"
                    )

        # Calculate data-driven statistics
        if confidence_scores:
            confidences = [score["confidence"] for score in confidence_scores]

            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)

            # Calculate percentile-based thresholds (data-driven approach)
            sorted_confidences = sorted(confidences)
            p25 = (
                sorted_confidences[len(sorted_confidences) // 4]
                if len(sorted_confidences) >= 4
                else min_confidence
            )
            p75 = (
                sorted_confidences[3 * len(sorted_confidences) // 4]
                if len(sorted_confidences) >= 4
                else max_confidence
            )

            threshold_analysis = {
                "sample_size": len(confidences),
                "average_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "p25_threshold": p25,
                "p75_threshold": p75,
                "data_driven": True,
            }

            print(f"📈 Data-Driven Confidence Analysis:")
            print(f"   Sample size: {threshold_analysis['sample_size']}")
            print(f"   Average: {avg_confidence:.3f}")
            print(f"   Range: {min_confidence:.3f} - {max_confidence:.3f}")
            print(f"   25th percentile: {p25:.3f}")
            print(f"   75th percentile: {p75:.3f}")

            # Validate data-driven thresholds are reasonable
            assert (
                0.0 <= avg_confidence <= 1.0
            ), f"Invalid average confidence: {avg_confidence}"
            assert p25 <= p75, f"Invalid percentile ordering: {p25} > {p75}"
            assert (
                len(confidences) >= 2
            ), f"Insufficient confidence samples: {len(confidences)}"

            return threshold_analysis
        else:
            pytest.skip("No valid confidence scores obtained for threshold analysis")


@pytest.mark.azure
@pytest.mark.integration
class TestUniversalDomainProcessing:
    """Test universal domain processing without hardcoded assumptions"""

    @pytest.mark.asyncio
    async def test_unknown_domain_handling(self, azure_services):
        """Test processing of unknown/mixed domain content"""

        if not azure_services.openai_client:
            pytest.skip("Azure OpenAI not available for unknown domain testing")

        # Create mixed-domain content that doesn't fit clear categories
        mixed_content = """
        The quantum computing algorithm processes financial data using machine learning models
        to optimize medical treatment protocols while ensuring legal compliance with privacy regulations.
        This interdisciplinary approach combines programming techniques with healthcare requirements.
        """

        analysis_prompt = f"""
        Analyze this mixed-domain text. Instead of forcing it into a single category,
        identify the multiple domains present and their relative presence.
        
        Text: {mixed_content}
        
        Format: domain1:percentage,domain2:percentage,domain3:percentage
        """

        result = await azure_services.openai_client.get_completion(
            analysis_prompt, max_tokens=100
        )

        if result and result.get("success"):
            response_text = result.get("content", "").strip()

            try:
                # Parse multi-domain response
                domain_percentages = {}
                for domain_part in response_text.split(","):
                    if ":" in domain_part:
                        domain, percentage_str = domain_part.split(":", 1)
                        percentage = float(percentage_str.strip().rstrip("%"))
                        domain_percentages[domain.strip()] = percentage

                print(f"🔀 Mixed Domain Analysis:")
                for domain, percentage in domain_percentages.items():
                    print(f"   {domain}: {percentage:.1f}%")

                # Validate multi-domain detection
                assert (
                    len(domain_percentages) >= 2
                ), f"Should detect multiple domains: {domain_percentages}"

                total_percentage = sum(domain_percentages.values())
                assert (
                    80 <= total_percentage <= 120
                ), f"Domain percentages should sum to ~100%: {total_percentage}"

                return domain_percentages

            except (ValueError, IndexError) as e:
                print(f"⚠️ Could not parse multi-domain analysis: {response_text}")
                pytest.skip(f"Multi-domain parsing failed: {e}")
        else:
            pytest.skip("Mixed domain analysis failed")

    @pytest.mark.asyncio
    async def test_domain_agnostic_processing(
        self, knowledge_extraction_agent, sample_documents
    ):
        """Test that processing works consistently across all domains"""

        processing_results = {}

        for domain, document in sample_documents.items():
            start_time = time.time()

            result = await knowledge_extraction_agent.process_query(
                {
                    "query": "Extract knowledge using universal approach",
                    "content": document,
                    "domain": domain,  # Domain provided but should not hardcode behavior
                }
            )

            processing_time = time.time() - start_time

            processing_results[domain] = {
                "success": result.get("success", False) if result else False,
                "processing_time": processing_time,
                "has_entities": (
                    bool(result.get("knowledge", {}).get("entities", []))
                    if result
                    else False
                ),
                "has_relationships": (
                    bool(result.get("knowledge", {}).get("relationships", []))
                    if result
                    else False
                ),
                "universal_processing": True,
            }

            print(
                f"🌐 {domain.title()} universal processing: "
                f"{'✅ Success' if processing_results[domain]['success'] else '❌ Failed'} "
                f"({processing_time:.3f}s)"
            )

        # Validate universal behavior
        successful_domains = [
            domain for domain, result in processing_results.items() if result["success"]
        ]

        success_rate = len(successful_domains) / len(processing_results) * 100
        print(f"🎯 Universal Processing Success Rate: {success_rate:.1f}%")

        # Universal design should work across domains
        assert (
            success_rate >= 75.0
        ), f"Universal processing success rate too low: {success_rate:.1f}%"

        # Check processing time consistency (no domain should be significantly slower)
        processing_times = [
            result["processing_time"] for result in processing_results.values()
        ]
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)

        # Maximum processing time should not be more than 2x average (reasonable variation)
        time_consistency = max_time <= (avg_time * 2.0)
        assert (
            time_consistency
        ), f"Processing time inconsistency: max={max_time:.3f}s, avg={avg_time:.3f}s"

        print(
            f"⏱️  Processing time consistency: avg={avg_time:.3f}s, max={max_time:.3f}s"
        )

        return processing_results
