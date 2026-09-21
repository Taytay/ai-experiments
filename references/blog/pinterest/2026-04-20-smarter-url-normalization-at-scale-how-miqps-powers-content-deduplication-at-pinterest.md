# Smarter URL Normalization at Scale: How MIQPS Powers Content Deduplication at Pinterest

- Source: https://medium.com/pinterest-engineering/smarter-url-normalization-at-scale-how-miqps-powers-content-deduplication-at-pinterest-4aa42e807d7d
- Publisher: Pinterest Engineering Blog (Medium)
- Published: 2026-04-20
- Authors: Shanhai Liao (Senior Software Engineer, Content Acquisition and Media Platform); Di Ruan (Senior Staff Software Engineer, Content Acquisition and Media Platform); Evan Li (Senior Engineering Manager, Content Acquisition and Media Platform)
- Captured: 2026-09-21 via Chrome page text extraction. Figures and tables were images and are noted in place.

---

## Introduction

Accurate content understanding underpins Pinterest's ability to drive distribution and engagement. This requires deep insight not just into the image itself, but also the outbound links or items to which those images point. At the foundation of this process lies a deceptively simple problem: URL normalization.

When Pinterest ingests content from millions of merchant domains, the same product page often appears under many different URLs. A single pair of shoes might be referenced by dozens of URL variations — each one decorated with different tracking parameters, session tokens, or analytics tags. While downstream systems can eventually deduplicate by content identity, the inability to recognize these duplicates at the URL level means every variation is independently fetched, rendered, and processed. At scale, this redundant ingestion and processing represents a significant waste of computational resources — rendering the same page dozens of times simply because its URLs differ in irrelevant parameters.

Item canonicalization — ensuring that identical items represented by different URLs are unified — is critical for organizing shopping catalogs and presenting a consistent experience to users. For many partners, a provided item ID determines canonical identity, but in its absence, the onus falls to advanced URL normalization to deduplicate effectively.

This post details the technical journey behind the Minimal Important Query Param Set (MIQPS) algorithm: a system that automatically learns which URL parameters matter for content identity, enabling dynamic and precise URL normalization at scale.

## Background: The URL Normalization Challenge

Consider a typical product URL from an e-commerce site:

```
https://example.com/shoes?id=42&color=red
```

This URL identifies a specific product variant. But in practice, the same product page is often reached through URLs like:

```
https://example.com/shoes?id=42&color=red&utm_source=facebook&session=abc123
https://example.com/shoes?id=42&color=red&ref=pinterest&click_id=xyz
https://example.com/shoes?id=42&color=red&tracking=campaign_spring
```

*Figure 1 (image): The URL duplication problem. Multiple URLs with different tracking parameters all resolve to the same product content.*

The parameters `utm_source`, `session`, `ref`, `click_id`, and `tracking` are all neutral - they don't change the content of the page. Meanwhile, `id` and `color` are non-neutral - they determine which product and variant are displayed.

The challenge is distinguishing between the two. For well-known e-commerce platforms, this can be solved with curated rules. Shopify URLs, for example, use variants as the key product differentiator. Salesforce Commerce Cloud uses parameters like `start`, `sz`, `prefn1`, and `prefv1`. For these platforms, static allowlists are sufficient.

But Pinterest ingests content from a large number of domains, operating on a wide variety of platforms.

For this long tail of domains, URL parameter conventions vary wildly. Static rules cannot scale to cover them all. We need a dynamic, data-driven approach.

## The MIQPS Algorithm

The core insight behind MIQPS is straightforward: if removing a query parameter changes the content of a page, that parameter is important; if it doesn't, the parameter is noise and can be safely stripped. Crucially, this analysis runs independently per domain — each merchant site gets its own MIQPS map, because the same parameter name can be meaningful on one domain and irrelevant on another.

The algorithm operates in three steps.

### Step 1: Collect the URL Corpus

As Pinterest's content ingestion pipeline processes URLs from domains, the system accumulates a corpus of observed URLs per domain. This corpus is stored durably and represents a snapshot of all the URL variations seen for a given domain. It serves as the input to the MIQPS analysis.

### Step 2: Group URLs by Query Parameter Pattern

Not all URLs from a domain share the same set of query parameters. A product page URL might carry `{id, color, utm_source}` while a category page might carry `{category, page, sort}`. Analyzing them together would be meaningless.

Moreover, the same parameter name can play different roles depending on its context. Consider the parameter `ref`: on a product page URL like `example.com/product?id=42&ref=homepage`, `ref` is purely a tracking parameter and is neutral - removing it doesn't change the product displayed. But on a comparison page URL like `example.com/compare?ref=99`, the same `ref` parameter identifies which items to compare and is non-neutral. By grouping URLs by their full parameter pattern, the algorithm evaluates each parameter within its specific context, correctly classifying it as neutral in one pattern and non-neutral in another.

To address this, the algorithm groups URLs by their query parameter pattern — the sorted set of parameter names present in the URL.

*(Image: example table of URLs grouped by query pattern.)*

URLs sharing the same query pattern are grouped together. The top K patterns by URL count are selected for analysis, focusing computational resources on the patterns that matter most.

### Step 3: For Each Pattern, Test Each Parameter

For each query parameter within a pattern, the algorithm determines whether it is neutral or non-neutral through empirical testing:

1. **Sample:** Select up to S URLs with distinct values for the parameter under test.
2. **Compare:** For each sampled URL, compute the content ID — a fingerprint derived from the page's rendered visual content — for both:
   - The original URL (with the parameter present)
   - A modified URL (with the parameter removed)
3. **Classify:** If removing the parameter changes the content ID in at least T% of samples, the parameter is classified as non-neutral (important). Otherwise, it is neutral (safe to drop).

The content ID is a hash of the page's visual representation, meaning two URLs that render the same visible content will produce the same content ID, even if their underlying HTML differs slightly. This particular fingerprinting approach leverages Pinterest's in-house page rendering infrastructure, which is tailored to our content pipeline. The core MIQPS algorithm, however, is agnostic to how the content fingerprint is produced — it only requires a function that returns the same identifier for the same page content. Third parties looking to adopt a similar approach could substitute alternatives such as DOM tree hashing, HTTP response body checksums, or even simpler heuristics like comparing the `<title>` and Open Graph metadata across URL variants. The key principle remains the same: compare some representation of the page content with and without each parameter to determine its importance.

A natural question is: why not simply use the **canonical URL** declared in the page's HTML (via the `<link rel="canonical">` tag) to resolve duplicates? If the merchant provides a canonical URL, two variant URLs pointing to the same product should share the same canonical, making deduplication trivial. In practice, however, canonical URLs are unreliable at scale. Many merchant sites omit them entirely, set them incorrectly (e.g., pointing every page to the homepage), or include tracking parameters in the canonical URL itself. Because we cannot assume canonical URLs are present or correct across the long tail of merchant domains, MIQPS uses visual content comparison as a ground-truth signal that works regardless of how well-maintained a site's metadata is.

### Algorithm Parameters

The behavior of the MIQPS algorithm is governed by a small set of tunable parameters:

*(Image: table of tunable parameters. From the text, these include K = number of top query patterns analyzed, S = max sample URLs per parameter, T = mismatch-rate threshold in percent, N = minimum successful tests, and A = anomaly-rate threshold in percent.)*

Two additional design choices make the algorithm practical at scale:

- **Early exit optimization:** If the mismatch rate already exceeds T% after N successful tests, we stop testing that parameter early. This avoids unnecessary page rendering calls for parameters that are clearly non-neutral.
- **Conservative default:** When fewer than N sample URLs are available for a parameter, it is treated as non-neutral by default. The system errs on the side of keeping parameters rather than dropping ones that might matter.

### Putting It Together

*Figure 2 (image): The MIQPS computation pipeline.*

The output of this pipeline is a MIQPS map: a mapping from each query parameter pattern to the set of non-neutral parameters within that pattern. This map is published to a configuration store and consumed at runtime during URL normalization.

## Multi-Layer Normalization Strategy

MIQPS does not operate in isolation. In production, URL normalization combines static rules with the dynamically computed MIQPS. Static rules capture known conventions — curated allowlists for recognized e-commerce platforms and regex patterns for widely used parameter naming schemes. These rules handle cases where we already have high confidence about which parameters matter.

MIQPS complements these static rules by covering the long tail of domains where no predefined rules exist. A URL parameter is kept if it is matched by either the static rules or the MIQPS non-neutral set. Only parameters that pass neither check are stripped. This combination ensures broad coverage: static rules provide immediate, reliable handling for known platforms, while MIQPS dynamically adapts to everything else.

## Anomaly Detection: Guarding Against Regressions

Computing MIQPS is inherently dependent on external page rendering. Pages can change, rendering infrastructure can have transient issues, and a domain's URL structure can shift between analysis runs. Without safeguards, a bad MIQPS computation could cause the system to start dropping parameters that are actually important — leading to content deduplication errors and degraded catalog quality.

To address this, the system includes an anomaly detection layer that compares each newly computed MIQPS against the previously published version. The comparison follows a set of conservative rules:

- **Parameter removed from non-neutral set (anomaly):** If a parameter that was previously classified as non-neutral is now classified as neutral, the pattern is flagged as anomalous. This is the dangerous case — it means we would start stripping a parameter that we previously determined was important.
- **Parameter added to non-neutral set (not anomalous):** If a previously neutral parameter is now classified as non-neutral, this is not considered an anomaly. It simply means we discovered a new important parameter, and the worst case is keeping slightly more parameters than necessary.
- **Pattern removed entirely (not anomalous):** If a query pattern from the previous MIQPS is absent in the new one, this is not flagged. Patterns can naturally disappear as a domain's URL structure evolves.

If more than A% of existing patterns are flagged as anomalous, the entire MIQPS update is rejected and the previous version is retained. This ensures the system never regresses — it errs on the side of over-keeping parameters rather than accidentally dropping ones that affect content identity.

## System Architecture and Integration

The MIQPS system fits into Pinterest's content processing pipeline as follows:

*Figure 3 (image): End-to-end system architecture. The content ingestion pipeline produces a URL corpus per domain. An offline job analyzes parameter importance via content ID comparison, then publishes the MIQPS to a config store after anomaly checks. The URL processor reads the MIQPS at runtime to normalize URLs during content processing.*

The architecture has three distinct phases:

1. **Content Ingestion:** As URLs are processed from domains, the system writes each unique URL to a per-domain corpus stored in S3. This happens continuously as part of normal content processing.
2. **MIQPS Computation:** After a content processing cycle completes for a domain, an offline job is triggered. This job downloads the URL corpus, runs the MIQPS algorithm (grouping, sampling, content ID comparison), performs anomaly detection, and publishes the result to both a config store (for runtime consumption) and S3 (for archival and debugging).
3. **URL Normalization:** At runtime, the URL processor loads the MIQPS map from the config store at initialization. For each URL it processes, it looks up the query pattern, retrieves the non-neutral parameter set, and strips all parameters not matched by any of the four normalization layers.

This separation of concerns means the expensive content ID comparison happens offline and asynchronously, while runtime URL normalization is a fast, in-memory lookup.

An alternative design would be to determine parameter importance **in realtime** — rendering the page with and without each parameter at the moment a URL is first encountered. This would eliminate staleness entirely and provide immediate coverage for newly discovered domains. However, we chose the offline approach for several reasons:

- **Latency:** Each content ID computation requires rendering a full page, which takes seconds. Testing every parameter in a URL would multiply this cost, adding unacceptable latency to the content processing pipeline.
- **Cost:** Offline analysis scales with the number of domains, while realtime analysis would scale with the number of URLs — orders of magnitude more expensive.
- **Reliability:** Transient rendering failures in an offline job are isolated and retryable. In a realtime path, they would directly block content processing.

In practice, the offline approach is a natural fit because URL parameter conventions change infrequently — on the order of weeks or months. The small amount of staleness between computation cycles is an acceptable tradeoff for the massive savings in cost, latency, and operational complexity.

## Conclusion

URL normalization may seem like a mundane infrastructure problem, but at Pinterest's scale — with a large number of domains and billions of URLs — getting it right has outsized impact on content quality.

The MIQPS algorithm brings several key properties to this challenge:

- **Dynamic and data-driven:** MIQPS automatically adapts to each domain's URL conventions without requiring manual configuration or domain-specific rules. As a domain's URL structure evolves, the algorithm discovers new patterns and adjusts accordingly.
- **Layered and defense-in-depth:** The multi-layer normalization strategy combines static allowlists, regex patterns, and dynamically computed MIQPS. Each layer catches a different class of parameters, and a parameter only needs to match one layer to be preserved.
- **Conservative and regression-resistant:** The anomaly detection system ensures that MIQPS updates never regress — previously important parameters cannot be silently dropped. The system consistently errs on the side of keeping parameters rather than stripping them.
- **Scalable and cost-efficient:** By grouping URLs by pattern, focusing on the top K patterns, and using early exit optimizations, the algorithm keeps computational costs manageable even across hundreds of thousands of domains.

By aligning normalization strategies with proven content identity signals, MIQPS ensures every unique item or experience is surfaced cleanly — improving search and recommendations, downstream catalog management, and ultimately the user experience.
