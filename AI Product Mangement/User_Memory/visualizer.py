import os
import json
import base64
from dataclasses import asdict
from models import TravelAgentProfile

def generate_html_visualizer(agents: list[TravelAgentProfile], interaction_logs: list, communications_logs: list, output_dir: str):
    agents_json = []
    for a in agents:
        # Filter logs for this specific agent
        agent_logs = [log for log in interaction_logs if log["agent_id"] == a.agent_id]
        agent_comms = [comm for comm in communications_logs if comm["agent_id"] == a.agent_id]
        
        mock_token = {
            "core_identity": {
                "origin_market": a.origin,
                "device": a.device,
                "login_status": a.login_status,
                "consent": a.consent_opt_in,
                "personalisation_id": a.personalisation_id if a.consent_opt_in else "ANONYMIZED"
            },
            "operational_preferences": {
                "baggage_tolerance": a.baggage_tolerance if a.consent_opt_in else "default",
                "proximity_anxiety": a.proximity_anxiety if a.consent_opt_in else "default",
                "preferred_alliance": a.preferred_alliance if a.consent_opt_in else "none"
            },
            "intent_token": {
                "active_destination": "LON" if a.origin[:3] in ["DEL", "BOM", "NYC", "LAX"] else "DEL",
                "trip_vibe": a.persona if a.consent_opt_in else "None",
                "flight_dates": "2026-07-15 to 2026-07-22" if a.consent_opt_in else ""
            }
        }
        token_bytes = json.dumps(mock_token).encode('utf-8')
        b64_header = base64.b64encode(token_bytes).decode('utf-8')
        
        agents_json.append({
            "profile": asdict(a),
            "token": mock_token,
            "header": f"X-Skyscanner-HCE-Token: {b64_header}",
            "interaction_logs": agent_logs,
            "communications_logs": agent_comms
        })
        
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyscanner HCE - Traveller Memory Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Mono&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Skyscanner Brand Palette & Layout Styling */
            --bg-color: #f1f2f8; /* clean grey-white background for real Skyscanner feel */
            --panel-bg: #ffffff; /* white panels */
            --border-color: #dcdfe6;
            --text-color: #070a25; /* deep navy text */
            --text-muted: #626573;
            --sky-blue: #0062e3; /* Skyscanner Brand Blue */
            --brand-green: #00a699; /* Skyscanner Brand Green */
            --brand-navy: #070a25; /* Skyscanner deep navy */
            --nudge-bg: rgba(0, 166, 153, 0.08);
            --nudge-border: rgba(0, 166, 153, 0.3);
            --nudge-text: #00877a;
            --control-red: #d32f2f;
            --card-shadow: 0 4px 12px rgba(7, 10, 37, 0.06);
            --hdr-grad: linear-gradient(135deg, #070a25, #0a1145);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 24px;
            min-height: 100vh;
        }
        
        header {
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }
        
        .logo-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--sky-blue), #00b2d6);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 22px;
            color: #ffffff;
        }
        
        h1 {
            font-size: 22px;
            font-weight: 700;
            color: var(--brand-navy);
            letter-spacing: -0.5px;
        }
        
        .badge-pm {
            font-size: 11px;
            background: rgba(0, 98, 227, 0.1);
            color: var(--sky-blue);
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .grid-container {
            display: grid;
            grid-template-columns: 360px 1fr 1fr;
            gap: 24px;
            height: calc(100vh - 120px);
        }
        
        /* TECHNICAL CONTROL CENTER - Sleek Dark Theme */
        .broker-panel {
            background: #0b111e;
            color: #f3f4f6;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .broker-panel .panel-header {
            color: #00b2d6;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 12px;
            font-weight:700; font-size:15px; text-transform:uppercase; margin-bottom:16px;
        }
        
        .broker-panel select {
            width: 100%;
            background: #1f2937;
            color: white;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            outline: none;
            font-size: 14px;
            margin-bottom: 16px;
        }
        
        .broker-panel .profile-row {
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
            font-size: 13px;
        }
        
        .broker-panel .profile-label {
            color: #9ca3af;
        }
        
        .broker-panel .profile-value {
            font-weight: 600;
        }
        
        .broker-panel .code-box {
            background: #05080f;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            padding: 10px;
            font-family: 'Space Mono', monospace;
            font-size: 10.5px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
            margin-top: 8px;
            color: #6ee7b7;
        }
        
        .broker-panel .code-label {
            font-size: 10px;
            color: #9ca3af;
            margin-top: 14px;
            display: block;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .broker-panel .switch-container {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        /* UI MOCKUPS - White Skyscanner Theme */
        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }
        
        .panel-header {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: var(--brand-navy);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 8px;
        }
        
        .scrollable-content {
            flex: 1;
            overflow-y: auto;
            padding-right: 4px;
        }
        
        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }
        
        .switch input { opacity: 0; width: 0; height: 0; }
        
        .slider {
            position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
            background-color: #374151; transition: .3s; border-radius: 24px;
        }
        
        .slider:before {
            position: absolute; content: ""; height: 16px; width: 16px; left: 4px; bottom: 4px;
            background-color: white; transition: .3s; border-radius: 50%;
        }
        
        input:checked + .slider {
            background-color: var(--brand-green);
        }
        
        input:checked + .slider:before {
            transform: translateX(20px);
        }
        
        .skyscanner-screen {
            background: #f4f5f8;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-size: 13px;
        }
        
        .skyscanner-header {
            background: var(--brand-navy);
            padding: 14px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: white;
        }
        
        .nav-tabs {
            display: flex;
            gap: 12px;
            padding: 0 16px;
            background: var(--brand-navy);
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .tab {
            padding: 10px 14px;
            font-weight: 600;
            font-size: 12px;
            cursor: pointer;
            color: #a3a6b5;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }
        
        .tab:hover { color: #ffffff; }
        
        .tab.active {
            color: #ffffff;
            border-bottom-color: var(--sky-blue);
        }
        
        /* Clean Skyscanner Search Widget */
        .search-widget-container {
            background: #ffffff;
            padding: 12px;
            border-bottom: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            margin-bottom: 12px;
        }
        
        .search-input-row {
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }
        
        .search-input-field {
            flex: 1;
            padding: 8px 10px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            font-size: 12px;
            font-weight: 600;
            color: var(--text-color);
            background: #fafafa;
        }
        
        .deal-card {
            background: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }
        
        .deal-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(7, 10, 37, 0.05);
        }
        
        .deal-card.highlighted {
            border-left: 4px solid var(--brand-green);
            background: #f7fdfb;
        }
        
        .deal-left {
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex: 1;
        }
        
        .deal-flight-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .airline-logo-lbl {
            font-weight: 700;
            color: var(--sky-blue);
            background: rgba(0, 98, 227, 0.05);
            padding: 4px 6px;
            border-radius: 4px;
            font-size: 11px;
            text-transform: uppercase;
        }
        
        .deal-time-line {
            font-size: 14px;
            font-weight: 700;
            color: var(--brand-navy);
        }
        
        .deal-stops-lbl {
            font-size: 11px;
            color: var(--text-muted);
            margin-left: 8px;
        }
        
        .deal-meta-specs {
            display: flex;
            gap: 12px;
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .deal-nudge {
            font-size: 11px;
            color: var(--brand-green);
            font-weight: 600;
            background: rgba(0, 166, 153, 0.07);
            padding: 6px 10px;
            border-radius: 4px;
            width: fit-content;
            border: 1px solid rgba(0, 166, 153, 0.15);
            margin-top: 4px;
            animation: pulseNudge 2s infinite ease-in-out;
        }
        
        @keyframes pulseNudge {
            0% { transform: scale(1); }
            50% { transform: scale(1.01); background: rgba(0, 166, 153, 0.1); }
            100% { transform: scale(1); }
        }
        
        .deal-right {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 6px;
            margin-left: 12px;
            min-width: 110px;
        }
        
        .price-badge {
            font-size: 20px;
            font-weight: 700;
            color: var(--brand-navy);
        }
        
        .deal-cta {
            background: var(--sky-blue);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            cursor: pointer;
            width: 100%;
            border: none;
            transition: background 0.15s;
        }
        
        .deal-cta:hover {
            background: #0050ba;
        }
        
        .deal-cta.upi-btn {
            background: var(--brand-green);
        }
        
        .deal-cta.upi-btn:hover {
            background: #00877d;
        }
        
        .alert-banner {
            background: rgba(211, 47, 47, 0.05);
            border: 1px dashed var(--control-red);
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 12px;
            font-size: 11.5px;
            color: var(--control-red);
            font-weight: 600;
        }
        
        .search-prefill-indicator {
            background: rgba(0, 166, 153, 0.06);
            color: #00877d;
            border: 1px solid rgba(0, 166, 153, 0.15);
            padding: 10px 12px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Telemetry Timeline Styling */
        .telemetry-timeline {
            margin-top: 10px;
            font-size: 11px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .telemetry-item {
            padding: 8px 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            line-height: 1.4;
        }
        .telemetry-step {
            color: #00b2d6;
            font-weight: 700;
            margin-right: 4px;
        }
        .telemetry-action {
            font-weight: 600;
            color: #e5e7eb;
        }
        .telemetry-desc {
            color: #9ca3af;
            font-family: 'Space Mono', monospace;
            font-size: 10px;
            margin-top: 4px;
        }
        .telemetry-comm {
            border-color: rgba(139, 92, 246, 0.3);
            background: rgba(139, 92, 246, 0.05);
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-title">
            <div class="logo">S</div>
            <div>
                <h1>Skyscanner HCE - Traveller Memory Visualizer</h1>
                <p style="color: var(--text-muted); font-size: 12px;">Centralized Horizontal Context Engine (A/B Test Verification Playground)</p>
            </div>
        </div>
        <span class="badge-pm">AI PM Evaluation Tool</span>
    </header>

    <div class="grid-container">
        <!-- technical dark control center -->
        <div class="broker-panel">
            <div class="panel-header">
                <span>1. Memory Engine Dashboard</span>
            </div>
            <div class="scrollable-content">
                <label style="font-size: 11px; color: #9ca3af; display:block; margin-bottom: 4px; font-weight:600; text-transform:uppercase;">Select Traveler Persona (N=30)</label>
                <select id="agent-selector" onchange="updateProfileView()">
                    __AGENT_OPTIONS__
                </select>
                
                <div class="switch-container">
                    <span style="font-size: 13px; font-weight:600; color:#f3f4f6;">HCE Consent Cookie Opt-In</span>
                    <label class="switch">
                        <input type="checkbox" id="consent-toggle" checked onchange="toggleConsent()">
                        <span class="slider"></span>
                    </label>
                </div>
                
                <div id="profile-fields"></div>
                
                <span class="code-label">X-Skyscanner-HCE-Token (API Header Contract)</span>
                <div class="code-box" id="header-box"></div>
                
                <span class="code-label">Context Broker Active Decoded Token</span>
                <div class="code-box" id="token-box" style="color: #6ee7b7; font-size: 10.5px; margin-bottom: 14px;"></div>

                <span class="code-label">Telemetry & Context Extraction Trace</span>
                <div id="telemetry-box" class="telemetry-timeline"></div>
            </div>
        </div>

        <!-- light-themed mockup - control -->
        <div class="panel">
            <div class="panel-header" style="color: var(--text-muted);">
                <span>2. Control Group (Generic Siloed UI)</span>
                <span style="font-size: 10px; padding: 2px 8px; background: #fafafa; border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-muted);">Siloed UI</span>
            </div>
            
            <div class="scrollable-content skyscanner-screen" id="control-screen">
                <div class="skyscanner-header">
                    <div style="font-weight: 700; color: #ffffff; font-size: 15px; display:flex; align-items:center; gap:6px;">
                        <span style="background:var(--sky-blue); color:white; border-radius:3px; padding: 1px 5px; font-size:11px;">S</span> Skyscanner
                    </div>
                    <div id="control-market-badge" style="font-size: 10px; color: #a3a6b5;">IN | INR</div>
                </div>
                
                <div class="nav-tabs">
                    <div class="tab active" id="control-tab-flights" onclick="setTab('Flights')">Flights</div>
                    <div class="tab" id="control-tab-stays" onclick="setTab('Stays')">Stays</div>
                    <div class="tab" id="control-tab-cars" onclick="setTab('Cars')">Car Hire</div>
                </div>
                
                <div style="padding: 16px; flex: 1; overflow-y: auto;" id="control-content"></div>
            </div>
        </div>

        <!-- light-themed mockup - treatment -->
        <div class="panel" style="border-color: rgba(0, 166, 153, 0.4);">
            <div class="panel-header" style="color: var(--brand-green);">
                <span>3. Treatment Group (HCE Dynamic UI)</span>
                <span style="font-size: 10px; padding: 2px 8px; background: rgba(0,166,153,0.06); border: 1px solid rgba(0,166,153,0.2); border-radius: 4px; color: var(--brand-green);">HCE Active</span>
            </div>
            
            <div class="scrollable-content skyscanner-screen" id="treatment-screen" style="border-color: rgba(0,166,153,0.2);">
                <div class="skyscanner-header">
                    <div style="font-weight: 700; color: #ffffff; font-size: 15px; display:flex; align-items:center; gap:6px;">
                        <span style="background:var(--brand-green); color:white; border-radius:3px; padding: 1px 5px; font-size:11px;">S</span> Skyscanner
                    </div>
                    <div id="treatment-market-badge" style="font-size: 10px; color: var(--brand-green); font-weight:600;">HCE ACTIVE</div>
                </div>
                
                <div class="nav-tabs">
                    <div class="tab active" id="treatment-tab-flights" onclick="setTab('Flights')">Flights</div>
                    <div class="tab" id="treatment-tab-stays" onclick="setTab('Stays')">Stays</div>
                    <div class="tab" id="treatment-tab-cars" onclick="setTab('Cars')">Car Hire</div>
                </div>
                
                <div style="padding: 16px; flex: 1; overflow-y: auto;" id="treatment-content"></div>
            </div>
        </div>
    </div>

    <script>
        const agents = __AGENTS_DATA__;
        let currentTab = 'Flights';
        let consentOverride = true;

        function toggleConsent() {
            consentOverride = document.getElementById('consent-toggle').checked;
            updateProfileView();
        }

        function setTab(tabName) {
            currentTab = tabName;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById('control-tab-' + tabName.toLowerCase()).classList.add('active');
            document.getElementById('treatment-tab-' + tabName.toLowerCase()).classList.add('active');
            updateProfileView();
        }

        function updateProfileView() {
            const agentId = document.getElementById('agent-selector').value;
            const data = agents.find(a => a.profile.agent_id === agentId);
            
            if (data.profile.consent_opt_in !== consentOverride) {
                data.token.core_identity.consent = consentOverride;
                data.token.core_identity.personalisation_id = consentOverride ? data.profile.personalisation_id : "ANONYMIZED";
                data.token.operational_preferences.baggage_tolerance = consentOverride ? data.profile.baggage_tolerance : "default";
                data.token.operational_preferences.proximity_anxiety = consentOverride ? data.profile.proximity_anxiety : "default";
                data.token.operational_preferences.preferred_alliance = consentOverride ? data.profile.preferred_alliance : "none";
                data.token.intent_token.trip_vibe = consentOverride ? data.profile.persona : "None";
                data.token.intent_token.flight_dates = consentOverride ? "2026-07-15 to 2026-07-22" : "";
                
                const tokenStr = JSON.stringify(data.token);
                const b64 = btoa(unescape(encodeURIComponent(tokenStr)));
                data.header = "X-Skyscanner-HCE-Token: " + b64;
            }

            const profile = data.profile;
            const token = data.token;
            
            document.getElementById('profile-fields').innerHTML = `
                <div class="profile-row"><span class="profile-label">Agent ID</span><span class="profile-value">${profile.agent_id}</span></div>
                <div class="profile-row"><span class="profile-label">Persona Archetype</span><span class="profile-value">${profile.persona}</span></div>
                <div class="profile-row"><span class="profile-label">Origin Market</span><span class="profile-value">${profile.origin} (${profile.country})</span></div>
                <div class="profile-row"><span class="profile-label">Device Type</span><span class="profile-value">${profile.device}</span></div>
                <div class="profile-row"><span class="profile-label">Login Status</span><span class="profile-value">${profile.login_status}</span></div>
                <div class="profile-row"><span class="profile-label">HCE Cookie State</span>
                    <span class="profile-value" style="color: ${token.core_identity.consent ? '#10b981' : '#ef4444'}; font-weight:700;">
                        ${token.core_identity.consent ? 'Stitched & Consented' : 'Opted-Out / Blocked'}
                    </span>
                </div>
            `;
            
            document.getElementById('header-box').innerText = data.header;
            document.getElementById('token-box').innerText = JSON.stringify(token, null, 2);
            
            // Render telemetry trace timeline
            renderTelemetry(data);
            
            renderControlScreen(profile, token);
            renderTreatmentScreen(profile, token);
        }

        function renderTelemetry(data) {
            const logs = data.interaction_logs || [];
            const comms = data.communications_logs || [];
            const container = document.getElementById('telemetry-box');
            
            if (logs.length === 0) {
                container.innerHTML = '<div style="color:#6b7280; font-style:italic; font-size:11px; margin-top:8px;">No telemetry logs recorded. Run simulation.py to compile history traces.</div>';
                return;
            }
            
            let combined = [];
            logs.forEach(l => {
                combined.push({
                    type: 'interaction',
                    step: l.step,
                    action: l.action,
                    state: l.state_entered,
                    desc: l.applied_ui_layout,
                    latency: l.api_latency_ms,
                    stitch: l.cookie_stitched
                });
            });
            
            comms.forEach(c => {
                combined.push({
                    type: 'communication',
                    step: c.step,
                    action: 'PRICE_DROP_ALERT_SENT',
                    desc: `Copy: "${c.payload}" | Clicked: ${c.clicked} | Redirected: ${c.completed_redirect}`,
                    latency: 'N/A',
                    stitch: 'N/A'
                });
            });
            
            combined.sort((a, b) => a.step - b.step);
            
            let html = '';
            combined.forEach(item => {
                const isComm = item.type === 'communication';
                const cssClass = isComm ? 'telemetry-item telemetry-comm' : 'telemetry-item';
                
                let actionText = item.action;
                if (actionText === 'SEARCH' && item.step === 1) {
                    actionText = 'INTENT_EXTRACTED_BY_LLM';
                } else if (actionText === 'REDIRECT_PARTNER') {
                    actionText = 'REDIRECT_PARTNER & MEMORY_EXTRACTED';
                }
                
                html += `
                    <div class="${cssClass}">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span class="telemetry-step">Step ${item.step}</span>
                            <span class="telemetry-action" style="color: ${isComm ? '#c084fc' : (actionText.includes('EXTRACTED') ? '#34d399' : '#e5e7eb')}">${actionText}</span>
                        </div>
                        <div class="telemetry-desc">${item.desc}</div>
                        ${item.latency !== 'N/A' ? `
                        <div style="font-size: 9px; color:#4b5563; margin-top:4px; font-family: monospace;">
                            Latency: ${item.latency}ms | Cookie Stitched: ${item.stitch}
                        </div>` : ''}
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function renderControlScreen(profile, token) {
            const content = document.getElementById('control-content');
            document.getElementById('control-market-badge').innerText = profile.origin + " | Siloed";
            
            const priceSymbol = profile.country === 'India' ? '₹' : (profile.country === 'UK' ? '£' : '$');
            const flightPriceVal1 = profile.country === 'India' ? '42,500' : (profile.country === 'UK' ? '410' : '520');
            const flightPriceVal2 = profile.country === 'India' ? '55,200' : (profile.country === 'UK' ? '530' : '680');
            
            if (currentTab === 'Flights') {
                content.innerHTML = `
                    <div class="alert-banner">
                        <span>⚠️ Generic Banner: Search and compare flights globally! Lowest rates guaranteed.</span>
                    </div>
                    
                    <div class="search-widget-container">
                        <div class="search-input-row">
                            <div class="search-input-field">${profile.origin}</div>
                            <div style="align-self:center; font-weight:700; color:var(--text-muted);">⇄</div>
                            <div class="search-input-field">Where to?</div>
                        </div>
                        <div class="search-input-row">
                            <div class="search-input-field">Select Dates</div>
                            <div class="search-input-field">1 Guest</div>
                        </div>
                    </div>
                    
                    <div class="deal-card">
                        <div class="deal-left">
                            <div class="deal-flight-row">
                                <span class="airline-logo-lbl">Indigo</span>
                                <span class="deal-time-line">08:00 → 14:00<span class="deal-stops-lbl">Direct</span></span>
                            </div>
                            <div class="deal-meta-specs">
                                <span>8h 30m</span>
                                <span>LCC Standard</span>
                            </div>
                        </div>
                        <div class="deal-right">
                            <span class="price-badge">${priceSymbol}${flightPriceVal1}</span>
                            <button class="deal-cta">Redirect</button>
                        </div>
                    </div>
                    
                    <div class="deal-card">
                        <div class="deal-left">
                            <div class="deal-flight-row">
                                <span class="airline-logo-lbl">BA</span>
                                <span class="deal-time-line">10:30 → 16:00<span class="deal-stops-lbl">Direct</span></span>
                            </div>
                            <div class="deal-meta-specs">
                                <span>9h 00m</span>
                                <span>oneworld carrier</span>
                            </div>
                        </div>
                        <div class="deal-right">
                            <span class="price-badge">${priceSymbol}${flightPriceVal2}</span>
                            <button class="deal-cta">Redirect</button>
                        </div>
                    </div>
                `;
            } else if (currentTab === 'Stays') {
                const hotelPrice = profile.country === 'India' ? '8,200' : (profile.country === 'UK' ? '80' : '110');
                content.innerHTML = `
                    <div class="search-widget-container">
                        <div class="search-input-row">
                            <input type="text" class="search-input-field" placeholder="Where do you want to stay?" style="background:#fff;">
                        </div>
                        <div class="search-input-row">
                            <input type="text" class="search-input-field" placeholder="Check-in" style="background:#fff;">
                            <input type="text" class="search-input-field" placeholder="Check-out" style="background:#fff;">
                        </div>
                    </div>
                    
                    <div class="deal-card">
                        <div class="deal-left">
                            <div style="font-weight: 700; font-size:14px;">Central Plaza Hotel</div>
                            <div class="deal-meta-specs">
                                <span>LON Central | ⭐ 4.1 Good</span>
                            </div>
                        </div>
                        <div class="deal-right">
                            <span class="price-badge">${priceSymbol}${hotelPrice}/night</span>
                            <button class="deal-cta">View Deal</button>
                        </div>
                    </div>
                `;
            } else {
                const carPrice = profile.country === 'India' ? '3,500' : (profile.country === 'UK' ? '35' : '48');
                content.innerHTML = `
                    <div class="deal-card">
                        <div class="deal-left">
                            <div style="font-weight: 700; font-size:14px;">Budget Compact Car</div>
                            <div class="deal-meta-specs">
                                <span>Hyundai i10 or similar | 4 seats | manual</span>
                            </div>
                        </div>
                        <div class="deal-right">
                            <span class="price-badge">${priceSymbol}${carPrice}/day</span>
                            <button class="deal-cta">View Car</button>
                        </div>
                    </div>
                `;
            }
        }

        function renderTreatmentScreen(profile, token) {
            const content = document.getElementById('treatment-content');
            const hceActive = token.core_identity.consent;
            
            document.getElementById('treatment-market-badge').innerHTML = hceActive 
                ? `<span style="color: var(--brand-green);">HCE PERSONALIZED ACTIVE</span>` 
                : `<span style="color: var(--text-muted);">HCE INACTIVE (NO CONSENT)</span>`;
            
            if (!hceActive) {
                renderControlScreen(profile, token);
                document.getElementById('treatment-content').innerHTML = document.getElementById('control-content').innerHTML;
                return;
            }

            const priceSymbol = profile.country === 'India' ? '₹' : (profile.country === 'UK' ? '£' : '$');
            const flightPriceVal1 = profile.country === 'India' ? '42,500' : (profile.country === 'UK' ? '410' : '520');
            const flightPriceVal2 = profile.country === 'India' ? '55,200' : (profile.country === 'UK' ? '530' : '680');
            const flightPriceVal3 = profile.country === 'India' ? '49,800' : (profile.country === 'UK' ? '480' : '610');
            
            const dest = token.intent_token.active_destination || "LON";
            const p = profile.persona;
            
            if (currentTab === 'Flights') {
                let layoutHtml = '';
                
                layoutHtml += `
                    <div class="search-widget-container">
                        <div class="search-input-row">
                            <div class="search-input-field">${profile.origin}</div>
                            <div style="align-self:center; font-weight:700; color:var(--brand-green);">⇄</div>
                            <div class="search-input-field">${dest}</div>
                        </div>
                        <div class="search-input-row">
                            <div class="search-input-field" style="border-color:var(--brand-green); background:#f0faf8;">July 15 - July 22</div>
                            <div class="search-input-field" style="border-color:var(--brand-green); background:#f0faf8;">
                                \${p === 'Multi-Gen Family Planner' ? '4 Passengers (Family)' : '1 Passenger'}
                            </div>
                        </div>
                    </div>
                `;
                
                if (p === 'Value Hacker') {
                    layoutHtml += `
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div class="deal-flight-row">
                                    <span class="airline-logo-lbl" style="background:rgba(0,166,153,0.1); color:var(--brand-green);">Indigo</span>
                                    <span class="deal-time-line">08:00 → 14:00<span class="deal-stops-lbl" style="color:var(--brand-green); font-weight:700;">Direct</span></span>
                                </div>
                                <div class="deal-meta-specs">
                                    <span>8h 30m</span>
                                    <span style="color:var(--brand-green); font-weight:600;">UPI Direct Pay Available</span>
                                </div>
                                <div class="deal-nudge">💼 Value Hacker: Extra 15kg checked + 7kg cabin luggage included in fare</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${flightPriceVal1}</span>
                                <button class="deal-cta upi-btn">Pay via UPI</button>
                            </div>
                        </div>
                    `;
                } else if (p === 'Loyalty Loyalist' || profile.preferred_alliance !== 'none') {
                    const allianceLabel = profile.preferred_alliance === 'star_alliance' ? 'Star Alliance' : (profile.preferred_alliance === 'oneworld' ? 'oneworld' : 'SkyTeam');
                    layoutHtml += `
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div class="deal-flight-row">
                                    <span class="airline-logo-lbl" style="background:rgba(0,166,153,0.1); color:var(--brand-green);">BA</span>
                                    <span class="deal-time-line">10:30 → 16:00<span class="deal-stops-lbl">Direct</span></span>
                                </div>
                                <div class="deal-meta-specs">
                                    <span>9h 00m</span>
                                    <span style="color:var(--brand-green); font-weight:600;">\${allianceLabel} Preferred Partner</span>
                                </div>
                                <div class="deal-nudge">⭐ Loyalty Loyalist: Earn 4,500 Avios Points + double credit card miles</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${flightPriceVal2}</span>
                                <button class="deal-cta">Book & Earn</button>
                            </div>
                        </div>
                    `;
                } else if (p === 'Multi-Gen Family Planner') {
                    layoutHtml += `
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div class="deal-flight-row">
                                    <span class="airline-logo-lbl" style="background:rgba(0,166,153,0.1); color:var(--brand-green);">Air India</span>
                                    <span class="deal-time-line">13:45 → 19:30<span class="deal-stops-lbl">Direct</span></span>
                                </div>
                                <div class="deal-meta-specs">
                                    <span>9h 15m</span>
                                    <span style="color:var(--brand-green); font-weight:600;">Family-optimized airline</span>
                                </div>
                                <div class="deal-nudge">👨‍👩‍👧‍👦 Family Planner: Seats together guaranteed + 3 items of check-in luggage</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${flightPriceVal3}</span>
                                <button class="deal-cta">Group Book</button>
                            </div>
                        </div>
                    `;
                } else {
                    layoutHtml += `
                        <div class="deal-card">
                            <div class="deal-left">
                                <div class="deal-flight-row">
                                    <span class="airline-logo-lbl">Indigo</span>
                                    <span class="deal-time-line">08:00 → 14:00</span>
                                </div>
                                <div class="deal-meta-specs">
                                    <span>8h 30m</span>
                                </div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${flightPriceVal1}</span>
                                <button class="deal-cta">Redirect</button>
                            </div>
                        </div>
                    `;
                }
                content.innerHTML = layoutHtml;
                
            } else if (currentTab === 'Stays') {
                const hotelPrice1 = profile.country === 'India' ? '14,500' : (profile.country === 'UK' ? '140' : '185');
                const hotelPrice2 = profile.country === 'India' ? '18,000' : (profile.country === 'UK' ? '175' : '230');
                const hostelPrice = profile.country === 'India' ? '2,800' : (profile.country === 'UK' ? '28' : '36');
                
                content.innerHTML = `
                    <div class="search-prefill-indicator">
                        ⚡ prefilled from HCE Intent Token: \${dest} | July 15 - July 22 (7 nights)
                    </div>
                    
                    \${ p === 'Multi-Gen Family Planner' ? \`
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">Airport Family Suites & Apartments</div>
                                <div class="deal-meta-specs">
                                    <span>LHR Proximity | ⭐ 4.6 Excellent</span>
                                </div>
                                <div class="deal-nudge">👪 Family Nudge: Multi-room flat with kitchen, kids play area & free transfer</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${hotelPrice1}/night</span>
                                <button class="deal-cta">View Deal</button>
                            </div>
                        </div>
                    \` : p === 'Business Bleisure' ? \`
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">Radisson Executive Business Suites</div>
                                <div class="deal-meta-specs">
                                    <span>Business Core | ⭐ 4.8 Exceptional</span>
                                </div>
                                <div class="deal-nudge">💻 Bleisure Nudge: In-room workspace desk + 1Gbps Wi-Fi + early 10AM check-in</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${hotelPrice2}/night</span>
                                <button class="deal-cta">View Deal</button>
                            </div>
                        </div>
                    \` : \`
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">YHA London Central Youth Hostel</div>
                                <div class="deal-meta-specs">
                                    <span>Central London | ⭐ 4.2 Good</span>
                                </div>
                                <div class="deal-nudge">🎒 Solo Explorer: Top-rated social hostel with shared lounge & budget pricing</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${hostelPrice}/night</span>
                                <button class="deal-cta">View Deal</button>
                            </div>
                        </div>
                    \`}
                `;
            } else {
                const suvPrice = profile.country === 'India' ? '5,200' : (profile.country === 'UK' ? '52' : '68');
                const compactPrice = profile.country === 'India' ? '3,100' : (profile.country === 'UK' ? '30' : '40');
                
                if (profile.proximity_anxiety === 'high') {
                    content.innerHTML = `
                        <div class="deal-card highlighted" style="border-left-color: var(--sky-blue);">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">Heathrow Express (Fast Rail Transfer)</div>
                                <div class="deal-meta-specs">
                                    <span>LHR Terminal 5 → Paddington | Departs every 15m</span>
                                </div>
                                <div class="deal-nudge" style="color:var(--sky-blue); background:rgba(0,98,227,0.05); border-color:rgba(0,98,227,0.15);">🚄 Anxiety Nudge: High airport anxiety score. Direct rail connection bypasses highway traffic.</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}2,200</span>
                                <button class="deal-cta">Book Tickets</button>
                            </div>
                        </div>
                    `;
                } else if (p === 'Multi-Gen Family Planner') {
                    content.innerHTML = `
                        <div class="deal-card highlighted">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">7-Seater Large Family SUV</div>
                                <div class="deal-meta-specs">
                                    <span>Hertz Terminal 5 Counter | Nissan X-Trail or similar</span>
                                </div>
                                <div class="deal-nudge">👶 Family Planner: 2 infant booster seats + extra luggage capacity preselected</div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${suvPrice}/day</span>
                                <button class="deal-cta">Book SUV</button>
                            </div>
                        </div>
                    `;
                } else {
                    content.innerHTML = `
                        <div class="deal-card">
                            <div class="deal-left">
                                <div style="font-weight: 700; font-size:14px; color:var(--brand-navy);">Hyundai i10 Eco-Compact</div>
                                <div class="deal-meta-specs">
                                    <span>Enterprise LHR Core | 4 seats | fuel-efficient</span>
                                </div>
                            </div>
                            <div class="deal-right">
                                <span class="price-badge">\${priceSymbol}\${compactPrice}/day</span>
                                <button class="deal-cta">Book Car</button>
                            </div>
                        </div>
                    `;
                }
            }
        }

        updateProfileView();
    </script>
</body>
</html>
"""
    
    agent_options = "".join([f'<option value="{a["profile"]["agent_id"]}">{a["profile"]["agent_id"]} ({a["profile"]["persona"]})</option>' for a in agents_json])
    agents_data = json.dumps(agents_json)
    
    html_content = html_content.replace("__AGENT_OPTIONS__", agent_options)
    html_content = html_content.replace("__AGENTS_DATA__", agents_data)

    html_path = os.path.join(output_dir, "ui_mockup_demo.html")
    with open(html_path, "w") as f:
        f.write(html_content)
        
    print(f"🖥️  Interactive visual HTML playground written to {html_path}")
