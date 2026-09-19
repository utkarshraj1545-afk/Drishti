from datetime import datetime

def generate_report(
    change_percent: float | None,
    classification: str,
    audience: str = "general",
    language: str = "en",
) -> dict:
    """
    Generate audience-specific and multilingual report
    from NDVI analysis.
    """

    pct = change_percent if change_percent is not None else 0.0

    # Risk Assessment
    if pct <= -30:
        risk_level = "High"
    elif pct <= -15:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    # Default English Content

    technical_summary = (
        f"Mean NDVI changed by {pct:.1f}% "
        f"indicating {classification}."
    )

    if classification == "vegetation loss":

        simple_summary = (
            "Vegetation health has reduced noticeably in the selected area."
        )

        recommendations = [
            "Inspect fields for drought stress.",
            "Check irrigation availability.",
            "Monitor vegetation over the next few weeks.",
        ]

    elif classification == "vegetation gain":

        simple_summary = (
            "Vegetation health has improved compared to the previous period."
        )

        recommendations = [
            "Continue current land management practices.",
            "Monitor vegetation growth regularly.",
        ]

    else:

        simple_summary = (
            "No significant vegetation changes were detected."
        )

        recommendations = [
            "Continue routine monitoring."
        ]

    # Audience Mode

    if audience == "farmer":

        if classification == "vegetation loss":

            simple_summary = (
                "Crop health appears weaker than before. "
                "Please inspect fields and irrigation sources."
            )

        elif classification == "vegetation gain":

            simple_summary = (
                "Crop health appears better than before."
            )

        else:

            simple_summary = (
                "No major crop health changes were detected."
            )

    elif audience == "researcher":

        simple_summary = (
            f"NDVI variation of {pct:.1f}% "
            f"indicates {classification}."
        )

    elif audience == "government":

        simple_summary = (
            f"The selected area shows {classification}. "
            f"Field verification may be considered if required."
        )

    # Hindi Translation Layer

    if language == "hi":

        report_title = "दृष्टि वनस्पति परिवर्तन रिपोर्ट"

        risk_map = {
            "High": "उच्च",
            "Moderate": "मध्यम",
            "Low": "कम",
        }

        risk_level = risk_map.get(risk_level, risk_level)

        if classification == "vegetation loss":

            technical_summary = (
                f"औसत NDVI में {pct:.1f}% की कमी दर्ज की गई।"
            )

            if audience == "farmer":

                simple_summary = (
                    "फसल या वनस्पति की स्थिति पहले की तुलना में कमजोर दिखाई दे रही है। "
                    "कृपया सिंचाई और खेत की स्थिति की जांच करें।"
                )

            elif audience == "researcher":

                simple_summary = (
                    f"NDVI में {pct:.1f}% परिवर्तन "
                    f"{classification} को दर्शाता है।"
                )

            elif audience == "government":

                simple_summary = (
                    "चयनित क्षेत्र में वनस्पति हानि देखी गई है। "
                    "आवश्यक होने पर क्षेत्रीय सत्यापन किया जा सकता है।"
                )

            else:

                simple_summary = (
                    "चयनित क्षेत्र में वनस्पति स्वास्थ्य में कमी देखी गई है।"
                )

            recommendations = [
                "सूखे या जल तनाव की जांच करें।",
                "सिंचाई की उपलब्धता सुनिश्चित करें।",
                "अगले कुछ सप्ताह तक क्षेत्र की निगरानी करें।",
            ]

        elif classification == "vegetation gain":

            technical_summary = (
                f"औसत NDVI में {pct:.1f}% की वृद्धि दर्ज की गई।"
            )

            simple_summary = (
                "वनस्पति की स्थिति पहले की तुलना में बेहतर दिखाई दे रही है।"
            )

            recommendations = [
                "वर्तमान कृषि प्रथाओं को जारी रखें।",
                "नियमित निगरानी बनाए रखें।",
            ]

        else:

            technical_summary = (
                f"औसत NDVI में {pct:.1f}% परिवर्तन दर्ज किया गया।"
            )

            simple_summary = (
                "कोई महत्वपूर्ण वनस्पति परिवर्तन नहीं पाया गया।"
            )

            recommendations = [
                "सामान्य निगरानी जारी रखें।"
            ]

    else:

        report_title = "Drishti Vegetation Change Report"

    return {
        "report_title": report_title,
        "generated_at": datetime.utcnow().isoformat(),
        "risk_level": risk_level,
        "technical_summary": technical_summary,
        "simple_summary": simple_summary,
        "recommendations": recommendations,
    }