//! ObelyZK Interactive TUI — fully interactive proving dashboard.
//!
//! Features:
//! - Tab modes: Monitor | Prove | Chat | On-Chain
//! - Interactive prompt input + model selector
//! - Real-time throughput sparklines
//! - Layer-by-layer proving progress
//! - Live GKR proof trace
//! - On-chain TX feed + verification status
//!
//! Color scheme: Cyan borders, Green accents, Pink highlights, Orange metrics.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        BarChart, Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Tabs, Wrap,
    },
    Frame,
};

// ── ObelyZK Color Palette ──────────────────────────────────────────────

const BG: Color = Color::Rgb(18, 18, 24);        // deep dark
const PANEL_BG: Color = Color::Rgb(24, 24, 32);   // slightly lighter
const BORDER: Color = Color::Rgb(0, 200, 200);    // cyan
const ACCENT: Color = Color::Rgb(100, 255, 120);  // green
const HIGHLIGHT: Color = Color::Rgb(255, 130, 200); // pink
const METRIC: Color = Color::Rgb(255, 180, 50);   // orange/amber
const DIM: Color = Color::Rgb(100, 100, 120);     // muted
const TEXT: Color = Color::Rgb(220, 220, 230);     // light text
const SUCCESS: Color = Color::Rgb(80, 255, 120);   // bright green
const WARN: Color = Color::Rgb(255, 200, 60);      // yellow
const ERROR: Color = Color::Rgb(255, 80, 80);      // red
const VIOLET: Color = Color::Rgb(160, 130, 255);   // purple
const CYAN_DIM: Color = Color::Rgb(0, 140, 140);   // muted cyan

// ── State ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Mode {
    Monitor,
    Prove,
    Chat,
    OnChain,
}

impl Mode {
    pub fn titles() -> Vec<&'static str> {
        vec!["1:Monitor", "2:Prove", "3:Chat", "4:On-Chain"]
    }
    pub fn index(&self) -> usize {
        match self {
            Mode::Monitor => 0,
            Mode::Prove => 1,
            Mode::Chat => 2,
            Mode::OnChain => 3,
        }
    }
}

/// Available models for selection.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub params: String,
    pub path: String,
    pub loaded: bool,
}

/// Live throughput sample for sparkline.
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    pub tok_per_sec: f64,
    pub timestamp_secs: f64,
}

/// A single layer proving event.
#[derive(Debug, Clone)]
pub struct LayerEvent {
    pub layer_idx: usize,
    pub layer_type: String,
    pub time_ms: u64,
    pub status: LayerStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerStatus {
    Pending,
    Proving,
    Done,
    Failed,
}

/// On-chain transaction record.
#[derive(Debug, Clone)]
pub struct OnChainTx {
    pub tx_hash: String,
    pub model: String,
    pub felts: usize,
    pub timestamp: String,
    pub verified: bool,
}

/// Chat message in the conversation.
#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub role: String,  // "user" or "assistant"
    pub content: String,
    pub proof_status: Option<String>,
    pub tx_hash: Option<String>,
}

/// Full interactive dashboard state.
pub struct InteractiveDashState {
    // Mode
    pub mode: Mode,

    // Model
    pub models: Vec<ModelEntry>,
    pub selected_model: usize,
    pub active_model: Option<String>,

    // GPU
    pub gpu_name: String,
    pub gpu_memory_gb: f32,
    pub gpu_utilization: f32,
    pub gpu_temp_c: f32,

    // Proving
    pub proving_active: bool,
    pub proving_layer: usize,
    pub proving_total_layers: usize,
    pub proving_matmul: usize,
    pub proving_total_matmuls: usize,
    pub proving_elapsed_secs: f64,
    pub layer_events: Vec<LayerEvent>,

    // Throughput sparkline (last 60 samples)
    pub throughput_history: Vec<u64>,
    pub current_tok_per_sec: f64,
    pub peak_tok_per_sec: f64,
    pub total_tokens_proven: usize,

    // STARK
    pub stark_compressing: bool,
    pub stark_time_secs: Option<f64>,
    pub stark_felts: Option<usize>,

    // On-chain
    pub on_chain_txs: Vec<OnChainTx>,
    pub verification_count: usize,
    pub contract_address: String,
    pub network: String,

    // Chat
    pub chat_messages: Vec<ChatMsg>,
    pub input_buffer: String,
    pub input_cursor: usize,

    // Runtime
    pub uptime_secs: u64,
    pub frame_count: u64,

    // Per-layer timing for bar chart (last proof)
    pub layer_timings_ms: Vec<(String, u64)>,
}

impl Default for InteractiveDashState {
    fn default() -> Self {
        Self {
            mode: Mode::Monitor,
            models: vec![
                ModelEntry {
                    name: "Qwen2.5-14B".into(), params: "14B".into(),
                    path: "/ephemeral/obelyzk-models/qwen2.5-14b".into(), loaded: true,
                },
                ModelEntry {
                    name: "GLM-4-9B".into(), params: "9B".into(),
                    path: "/ephemeral/obelyzk-models/glm-4-9b".into(), loaded: true,
                },
                ModelEntry {
                    name: "SmolLM2-135M".into(), params: "135M".into(),
                    path: "~/.obelyzk/models/smollm2-135m".into(), loaded: false,
                },
            ],
            selected_model: 0,
            active_model: Some("Qwen2.5-14B".into()),
            gpu_name: "NVIDIA H100 PCIe".into(),
            gpu_memory_gb: 80.0,
            gpu_utilization: 0.0,
            gpu_temp_c: 36.0,
            proving_active: false,
            proving_layer: 0,
            proving_total_layers: 337,
            proving_matmul: 0,
            proving_total_matmuls: 192,
            proving_elapsed_secs: 0.0,
            layer_events: Vec::new(),
            throughput_history: vec![0; 60],
            current_tok_per_sec: 0.0,
            peak_tok_per_sec: 0.0,
            total_tokens_proven: 0,
            stark_compressing: false,
            stark_time_secs: None,
            stark_felts: None,
            on_chain_txs: Vec::new(),
            verification_count: 0,
            contract_address: "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7".into(),
            network: "Starknet Sepolia".into(),
            chat_messages: Vec::new(),
            input_buffer: String::new(),
            input_cursor: 0,
            uptime_secs: 0,
            frame_count: 0,
            layer_timings_ms: Vec::new(),
        }
    }
}

// ── Main Render ────────────────────────────────────────────────────────

pub fn render_interactive(frame: &mut Frame, state: &InteractiveDashState) {
    let area = frame.area();

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Header + tabs
            Constraint::Length(3),   // Model bar + GPU status
            Constraint::Min(10),    // Main content (mode-dependent)
            Constraint::Length(3),   // Status bar / input
        ])
        .split(area);

    render_header_tabs(frame, outer[0], state);
    render_model_bar(frame, outer[1], state);

    match state.mode {
        Mode::Monitor => render_monitor(frame, outer[2], state),
        Mode::Prove => render_prove(frame, outer[2], state),
        Mode::Chat => render_chat(frame, outer[2], state),
        Mode::OnChain => render_onchain(frame, outer[2], state),
    }

    render_status_bar(frame, outer[3], state);
}

// ── Header + Tabs ──────────────────────────────────────────────────────

fn render_header_tabs(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(25), Constraint::Min(20)])
        .split(area);

    // Logo
    let logo = Paragraph::new(Line::from(vec![
        Span::styled(" ObelyZK ", Style::default().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(" v0.4.0 ", Style::default().fg(DIM)),
    ])).block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(BORDER)));
    frame.render_widget(logo, chunks[0]);

    // Tabs
    let titles: Vec<Line> = Mode::titles().iter().map(|t| Line::from(*t)).collect();
    let tabs = Tabs::new(titles)
        .select(state.mode.index())
        .highlight_style(Style::default().fg(BG).bg(HIGHLIGHT).add_modifier(Modifier::BOLD))
        .style(Style::default().fg(DIM))
        .divider(Span::styled(" │ ", Style::default().fg(CYAN_DIM)))
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(BORDER)));
    frame.render_widget(tabs, chunks[1]);
}

// ── Model Bar ──────────────────────────────────────────────────────────

fn render_model_bar(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Active model
    let model_name = state.active_model.as_deref().unwrap_or("none");
    let model_line = Line::from(vec![
        Span::styled(" MODEL ", Style::default().fg(BG).bg(VIOLET).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", model_name), Style::default().fg(TEXT).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} layers ", state.proving_total_layers), Style::default().fg(DIM)),
        Span::styled(format!(" {} matmuls ", state.proving_total_matmuls), Style::default().fg(DIM)),
    ]);
    frame.render_widget(
        Paragraph::new(model_line).block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(CYAN_DIM))),
        chunks[0],
    );

    // GPU status
    let gpu_line = Line::from(vec![
        Span::styled(" GPU ", Style::default().fg(BG).bg(METRIC).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", state.gpu_name), Style::default().fg(TEXT)),
        Span::styled(format!(" {:.0}GB ", state.gpu_memory_gb), Style::default().fg(DIM)),
        Span::styled(format!(" {:.0}°C ", state.gpu_temp_c), Style::default().fg(
            if state.gpu_temp_c > 80.0 { ERROR } else if state.gpu_temp_c > 60.0 { WARN } else { SUCCESS }
        )),
    ]);
    frame.render_widget(
        Paragraph::new(gpu_line).block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(CYAN_DIM))),
        chunks[1],
    );
}

// ── Mode: Monitor ──────────────────────────────────────────────────────

fn render_monitor(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),   // Throughput sparkline + stats
            Constraint::Length(6),   // Layer timing bar chart
            Constraint::Min(4),     // Recent events / log
        ])
        .split(area);

    // Throughput sparkline
    let spark_block = Block::default()
        .title(Span::styled(" Throughput (tok/s) ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER));

    let sparkline = Sparkline::default()
        .block(spark_block)
        .data(&state.throughput_history)
        .style(Style::default().fg(ACCENT))
        .bar_set(symbols::bar::NINE_LEVELS);
    frame.render_widget(sparkline, chunks[0]);

    // Layer timing bar chart
    let bar_data: Vec<(&str, u64)> = state.layer_timings_ms.iter()
        .take(20)
        .map(|(name, ms)| (name.as_str(), *ms))
        .collect();

    let barchart = BarChart::default()
        .block(Block::default()
            .title(Span::styled(" Layer Timings (ms) ", Style::default().fg(METRIC).add_modifier(Modifier::BOLD)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER)))
        .data(&bar_data)
        .bar_width(3)
        .bar_gap(1)
        .bar_style(Style::default().fg(METRIC))
        .value_style(Style::default().fg(DIM));
    frame.render_widget(barchart, chunks[1]);

    // Stats + recent events
    let stats = vec![
        Line::from(vec![
            Span::styled(" proven  ", Style::default().fg(DIM)),
            Span::styled(format!("{:.1} tok/s", state.current_tok_per_sec), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled("  peak ", Style::default().fg(DIM)),
            Span::styled(format!("{:.1}", state.peak_tok_per_sec), Style::default().fg(METRIC)),
            Span::styled("  total ", Style::default().fg(DIM)),
            Span::styled(format!("{}", state.total_tokens_proven), Style::default().fg(TEXT)),
            Span::styled("  on-chain ", Style::default().fg(DIM)),
            Span::styled(format!("{}", state.verification_count), Style::default().fg(VIOLET)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(stats)
            .block(Block::default()
                .title(Span::styled(" Stats ", Style::default().fg(TEXT).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[2],
    );
}

// ── Mode: Prove ────────────────────────────────────────────────────────

fn render_prove(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Progress gauge
            Constraint::Length(5),   // STARK compression
            Constraint::Min(4),     // Live proof trace
        ])
        .split(area);

    // GKR progress gauge
    let progress = if state.proving_total_layers > 0 {
        state.proving_layer as f64 / state.proving_total_layers as f64
    } else {
        0.0
    };
    let gauge_label = if state.proving_active {
        format!(
            "GKR Layer {}/{} · Matmul {}/{} · {:.1}s",
            state.proving_layer, state.proving_total_layers,
            state.proving_matmul, state.proving_total_matmuls,
            state.proving_elapsed_secs,
        )
    } else {
        "Idle — press Enter to prove".into()
    };

    let gauge = Gauge::default()
        .block(Block::default()
            .title(Span::styled(" GKR Proof Progress ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BORDER)))
        .gauge_style(Style::default().fg(ACCENT).bg(PANEL_BG))
        .ratio(progress)
        .label(Span::styled(gauge_label, Style::default().fg(TEXT)));
    frame.render_widget(gauge, chunks[0]);

    // STARK compression status
    let stark_lines = if state.stark_compressing {
        vec![
            Line::from(Span::styled(" Compressing GKR → Recursive STARK...", Style::default().fg(HIGHLIGHT))),
        ]
    } else if let Some(time) = state.stark_time_secs {
        vec![
            Line::from(vec![
                Span::styled(" STARK ", Style::default().fg(BG).bg(SUCCESS).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" {:.1}s ", time), Style::default().fg(METRIC).add_modifier(Modifier::BOLD)),
                Span::styled(format!(" {} felts ", state.stark_felts.unwrap_or(0)), Style::default().fg(TEXT)),
                Span::styled(" on-chain ready ", Style::default().fg(SUCCESS)),
            ]),
        ]
    } else {
        vec![
            Line::from(Span::styled(" Waiting for GKR proof to complete...", Style::default().fg(DIM))),
        ]
    };
    frame.render_widget(
        Paragraph::new(stark_lines)
            .block(Block::default()
                .title(Span::styled(" Recursive STARK ", Style::default().fg(HIGHLIGHT).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[1],
    );

    // Live proof trace
    let trace_items: Vec<ListItem> = state.layer_events.iter().rev().take(20).map(|e| {
        let (symbol, color) = match e.status {
            LayerStatus::Done => ("✓", SUCCESS),
            LayerStatus::Proving => ("▸", METRIC),
            LayerStatus::Pending => ("·", DIM),
            LayerStatus::Failed => ("✗", ERROR),
        };
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {} ", symbol), Style::default().fg(color)),
            Span::styled(format!("L{:<3}", e.layer_idx), Style::default().fg(DIM)),
            Span::styled(format!(" {:<12}", e.layer_type), Style::default().fg(TEXT)),
            Span::styled(format!(" {}ms", e.time_ms), Style::default().fg(METRIC)),
        ]))
    }).collect();

    frame.render_widget(
        List::new(trace_items)
            .block(Block::default()
                .title(Span::styled(" Live Proof Trace ", Style::default().fg(VIOLET).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[2],
    );
}

// ── Mode: Chat ─────────────────────────────────────────────────────────

fn render_chat(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),     // Chat messages
            Constraint::Length(3),   // Input box
        ])
        .split(area);

    // Messages
    let msg_items: Vec<ListItem> = state.chat_messages.iter().map(|m| {
        let (prefix, color) = if m.role == "user" {
            (" YOU ", ACCENT)
        } else {
            (" AI  ", HIGHLIGHT)
        };
        let mut spans = vec![
            Span::styled(prefix, Style::default().fg(BG).bg(color).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {}", m.content), Style::default().fg(TEXT)),
        ];
        if let Some(ref tx) = m.tx_hash {
            spans.push(Span::styled(format!("  TX: {}...", &tx[..18.min(tx.len())]), Style::default().fg(VIOLET)));
        }
        if let Some(ref status) = m.proof_status {
            spans.push(Span::styled(format!("  [{}]", status), Style::default().fg(
                if status == "verified" { SUCCESS } else { METRIC }
            )));
        }
        ListItem::new(Line::from(spans))
    }).collect();

    frame.render_widget(
        List::new(msg_items)
            .block(Block::default()
                .title(Span::styled(" Verifiable Conversation ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[0],
    );

    // Input box
    let input = Paragraph::new(Line::from(vec![
        Span::styled(" > ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(&state.input_buffer, Style::default().fg(TEXT)),
        Span::styled("█", Style::default().fg(ACCENT)),  // cursor
    ]))
    .block(Block::default()
        .title(Span::styled(" Type a message (Enter to send, Esc to cancel) ", Style::default().fg(DIM)))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(HIGHLIGHT)));
    frame.render_widget(input, chunks[1]);
}

// ── Mode: On-Chain ─────────────────────────────────────────────────────

fn render_onchain(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),   // Contract info
            Constraint::Min(6),     // TX list
        ])
        .split(area);

    // Contract info
    let contract_short = if state.contract_address.len() > 30 {
        format!("{}...{}", &state.contract_address[..16], &state.contract_address[state.contract_address.len()-8..])
    } else {
        state.contract_address.clone()
    };
    let info_lines = vec![
        Line::from(vec![
            Span::styled(" Contract ", Style::default().fg(BG).bg(VIOLET).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {} ", contract_short), Style::default().fg(TEXT)),
        ]),
        Line::from(vec![
            Span::styled(" Network  ", Style::default().fg(DIM)),
            Span::styled(&state.network, Style::default().fg(VIOLET)),
            Span::styled("  Verifications  ", Style::default().fg(DIM)),
            Span::styled(format!("{}", state.verification_count), Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(info_lines)
            .block(Block::default()
                .title(Span::styled(" Starknet Verification ", Style::default().fg(VIOLET).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[0],
    );

    // TX list
    let tx_items: Vec<ListItem> = state.on_chain_txs.iter().rev().map(|tx| {
        let tx_short = if tx.tx_hash.len() > 20 {
            format!("{}...{}", &tx.tx_hash[..10], &tx.tx_hash[tx.tx_hash.len()-6..])
        } else {
            tx.tx_hash.clone()
        };
        let status_style = if tx.verified {
            Style::default().fg(SUCCESS)
        } else {
            Style::default().fg(WARN)
        };
        ListItem::new(Line::from(vec![
            Span::styled(if tx.verified { " ✓ " } else { " ▸ " }, status_style),
            Span::styled(tx_short, Style::default().fg(VIOLET)),
            Span::styled(format!("  {} ", tx.model), Style::default().fg(TEXT)),
            Span::styled(format!(" {} felts ", tx.felts), Style::default().fg(METRIC)),
            Span::styled(format!(" {} ", tx.timestamp), Style::default().fg(DIM)),
        ]))
    }).collect();

    frame.render_widget(
        List::new(tx_items)
            .block(Block::default()
                .title(Span::styled(" Verified Proofs ", Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER))),
        chunks[1],
    );
}

// ── Status Bar ─────────────────────────────────────────────────────────

fn render_status_bar(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let uptime = format_uptime(state.uptime_secs);
    let status_color = if state.proving_active { METRIC } else { SUCCESS };
    let status_text = if state.proving_active { "PROVING" } else { "READY" };

    let bar = Line::from(vec![
        Span::styled(" ObelyZK ", Style::default().fg(BG).bg(ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", status_text), Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", uptime), Style::default().fg(METRIC)),
        Span::styled(format!(" {} ", state.network), Style::default().fg(VIOLET)),
        Span::styled(format!(" {} TXs ", state.verification_count), Style::default().fg(DIM)),
        Span::styled("  ", Style::default()),
        Span::styled("1-4", Style::default().fg(ACCENT)),
        Span::styled(" mode ", Style::default().fg(DIM)),
        Span::styled("q", Style::default().fg(ACCENT)),
        Span::styled(" quit ", Style::default().fg(DIM)),
        Span::styled("Enter", Style::default().fg(ACCENT)),
        Span::styled(" prove ", Style::default().fg(DIM)),
        Span::styled("Tab", Style::default().fg(ACCENT)),
        Span::styled(" model ", Style::default().fg(DIM)),
    ]);

    frame.render_widget(
        Paragraph::new(bar)
            .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(BORDER))),
        area,
    );
}

fn format_uptime(secs: u64) -> String {
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{}:{:02}:{:02}", h, m, s)
}
