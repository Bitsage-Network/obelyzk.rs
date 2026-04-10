//! ObelyZK Interactive TUI — live proving dashboard with real-time visualizations.
//!
//! Inspired by Couture Trace's aesthetic but built for ML proving:
//! - Custom sparklines (▁▂▃▄▅▆▇█) with rolling history
//! - Custom gauges (█·) for layer-by-layer progress
//! - Badge pills for status indicators
//! - 4 interactive modes: Monitor / Prove / Chat / On-Chain
//! - Background proving via mpsc channels
//! - Pulse animation on active operations
//!
//! Uses indexed terminal colors (not RGB) for SSH compatibility.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Wrap},
    Frame,
};

// ── Theme: ObelyZK Neon ────────────────────────────────────────────────
// All indexed colors for SSH/tmux compatibility.

pub struct Theme {
    pub bg: Color,
    pub panel: Color,
    pub border: Color,
    pub text: Color,
    pub muted: Color,
    pub accent: Color,        // primary action color (green)
    pub accent_soft: Color,   // dimmer accent
    pub highlight: Color,     // attention/pink
    pub metric: Color,        // numbers/orange
    pub success: Color,
    pub danger: Color,
    pub violet: Color,
    pub cyan: Color,
}

const NEON: Theme = Theme {
    bg:          Color::Indexed(234),   // very dark grey
    panel:       Color::Indexed(236),   // slightly lighter
    border:      Color::Indexed(44),    // cyan
    text:        Color::Indexed(252),   // light grey
    muted:       Color::Indexed(243),   // mid grey
    accent:      Color::Indexed(118),   // bright green
    accent_soft: Color::Indexed(70),    // dim green
    highlight:   Color::Indexed(213),   // pink
    metric:      Color::Indexed(208),   // orange
    success:     Color::Indexed(48),    // emerald
    danger:      Color::Indexed(196),   // red
    violet:      Color::Indexed(141),   // purple
    cyan:        Color::Indexed(51),    // bright cyan
};

// Shorthand
const T: &Theme = &NEON;

// ── Block elements for custom widgets ──────────────────────────────────

const SPARK_CHARS: &[char] = &[' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const GAUGE_FULL: &str = "█";
const GAUGE_EMPTY: &str = "░";
const PULSE_CHARS: &[&str] = &["◢", "◣", "◤", "◥"];

const HISTORY_LIMIT: usize = 80;

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
        match self { Mode::Monitor => 0, Mode::Prove => 1, Mode::Chat => 2, Mode::OnChain => 3 }
    }
    pub fn next(&self) -> Self {
        match self { Mode::Monitor => Mode::Prove, Mode::Prove => Mode::Chat, Mode::Chat => Mode::OnChain, Mode::OnChain => Mode::Monitor }
    }
}

#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub name: String,
    pub params: String,
    pub layers: usize,
    pub matmuls: usize,
    pub loaded: bool,
}

#[derive(Debug, Clone)]
pub struct LayerEvent {
    pub layer_idx: usize,
    pub layer_type: String,
    pub time_ms: u64,
    pub done: bool,
}

#[derive(Debug, Clone)]
pub struct OnChainTx {
    pub tx_hash: String,
    pub model: String,
    pub felts: usize,
    pub verified: bool,
}

#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
    pub proof_status: Option<String>,
    pub tx_hash: Option<String>,
}

pub struct InteractiveDashState {
    pub mode: Mode,

    // Models
    pub models: Vec<ModelEntry>,
    pub selected_model: usize,

    // GPU
    pub gpu_name: String,
    pub gpu_memory_gb: f32,
    pub gpu_temp_c: f32,
    pub gpu_util_pct: f32,

    // Proving state
    pub proving_active: bool,
    pub proving_layer: usize,
    pub proving_total_layers: usize,
    pub proving_matmul: usize,
    pub proving_total_matmuls: usize,
    pub proving_elapsed_secs: f64,
    pub proving_phase: String,   // "forward", "gkr", "stark", "on-chain", "idle"

    // History (rolling buffers for sparklines)
    pub throughput_history: Vec<u64>,
    pub gpu_util_history: Vec<u64>,
    pub layer_time_history: Vec<u64>,

    // Stats
    pub current_tok_per_sec: f64,
    pub peak_tok_per_sec: f64,
    pub total_tokens_proven: usize,
    pub total_proofs: usize,

    // STARK
    pub stark_time_secs: Option<f64>,
    pub stark_felts: Option<usize>,

    // On-chain
    pub on_chain_txs: Vec<OnChainTx>,
    pub verification_count: usize,
    pub contract: String,
    pub network: String,

    // Chat
    pub chat_messages: Vec<ChatMsg>,
    pub input_buffer: String,
    pub input_cursor: usize,

    // Layer trace (recent events)
    pub layer_events: Vec<LayerEvent>,

    // Runtime
    pub uptime_secs: u64,
    pub tick: u64,
}

impl Default for InteractiveDashState {
    fn default() -> Self {
        // Seed with sample data so it never looks empty
        let mut throughput = vec![0u64; HISTORY_LIMIT];
        let mut gpu_util = vec![0u64; HISTORY_LIMIT];
        let mut layer_time = vec![0u64; HISTORY_LIMIT];
        // Seed some initial values to show the sparkline shape
        for i in 0..HISTORY_LIMIT {
            throughput[i] = ((i as f64 * 0.15).sin().abs() * 5.0) as u64;
            gpu_util[i] = 20 + ((i as f64 * 0.1).sin().abs() * 60.0) as u64;
            layer_time[i] = 100 + ((i as f64 * 0.2).cos().abs() * 400.0) as u64;
        }

        Self {
            mode: Mode::Monitor,
            models: vec![
                ModelEntry { name: "Qwen2.5-14B".into(), params: "14B".into(), layers: 337, matmuls: 192, loaded: true },
                ModelEntry { name: "GLM-4-9B".into(), params: "9B".into(), layers: 281, matmuls: 160, loaded: true },
                ModelEntry { name: "SmolLM2-135M".into(), params: "135M".into(), layers: 25, matmuls: 12, loaded: false },
            ],
            selected_model: 0,
            gpu_name: "NVIDIA H100 PCIe".into(),
            gpu_memory_gb: 80.0,
            gpu_temp_c: 38.0,
            gpu_util_pct: 0.0,
            proving_active: false,
            proving_layer: 0,
            proving_total_layers: 337,
            proving_matmul: 0,
            proving_total_matmuls: 192,
            proving_elapsed_secs: 0.0,
            proving_phase: "idle".into(),
            throughput_history: throughput,
            gpu_util_history: gpu_util,
            layer_time_history: layer_time,
            current_tok_per_sec: 0.0,
            peak_tok_per_sec: 0.23,
            total_tokens_proven: 0,
            total_proofs: 0,
            stark_time_secs: None,
            stark_felts: None,
            on_chain_txs: vec![
                OnChainTx { tx_hash: "0x5ce1b41815e29a7b3dd0..".into(), model: "Qwen2.5-14B".into(), felts: 946, verified: true },
                OnChainTx { tx_hash: "0x542960d703a62d4beaac..".into(), model: "GLM-4-9B".into(), felts: 929, verified: true },
            ],
            verification_count: 7,
            contract: "0x1c208a5fe731c0d03b098b524f274c537587ea1d43d903838cc4a2bf90c40c7".into(),
            network: "Starknet Sepolia".into(),
            chat_messages: Vec::new(),
            input_buffer: String::new(),
            input_cursor: 0,
            layer_events: vec![
                LayerEvent { layer_idx: 336, layer_type: "RMSNorm".into(), time_ms: 12, done: true },
                LayerEvent { layer_idx: 335, layer_type: "MatMul".into(), time_ms: 245, done: true },
                LayerEvent { layer_idx: 334, layer_type: "Add".into(), time_ms: 8, done: true },
                LayerEvent { layer_idx: 333, layer_type: "MatMul".into(), time_ms: 231, done: true },
                LayerEvent { layer_idx: 332, layer_type: "RMSNorm".into(), time_ms: 11, done: true },
                LayerEvent { layer_idx: 331, layer_type: "MatMul".into(), time_ms: 252, done: true },
            ],
            uptime_secs: 0,
            tick: 0,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Consistent panel block with themed borders.
fn panel(title: &str, color: Color) -> Block<'_> {
    Block::default()
        .title(Span::styled(
            format!(" {} ", title),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(T.border))
}

/// Badge pill: colored text on colored bg.
fn badge(text: &str, fg: Color, bg: Color) -> Span<'static> {
    Span::styled(
        format!(" {} ", text),
        Style::default().fg(fg).bg(bg).add_modifier(Modifier::BOLD),
    )
}

/// Build a sparkline string from data using block characters.
fn sparkline_str(data: &[u64], width: usize) -> String {
    if data.is_empty() { return " ".repeat(width); }
    let max = data.iter().copied().max().unwrap_or(1).max(1);
    let start = if data.len() > width { data.len() - width } else { 0 };
    let slice = &data[start..];
    let mut s = String::with_capacity(width);
    for &v in slice {
        let idx = ((v as f64 / max as f64) * 8.0).round() as usize;
        s.push(SPARK_CHARS[idx.min(8)]);
    }
    // Pad if shorter
    while s.len() < width { s.push(' '); }
    s
}

/// Build a gauge line: filled + empty blocks.
fn gauge_line(ratio: f64, width: usize, fill_color: Color, empty_color: Color) -> Vec<Span<'static>> {
    let filled = (ratio * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);
    vec![
        Span::styled(GAUGE_FULL.repeat(filled), Style::default().fg(fill_color)),
        Span::styled(GAUGE_EMPTY.repeat(empty), Style::default().fg(empty_color)),
    ]
}

/// Format uptime as H:MM:SS.
/// Ring-buffer push: remove oldest when at capacity.
pub fn push_limited(buf: &mut Vec<u64>, val: u64) {
    if buf.len() >= HISTORY_LIMIT {
        buf.remove(0);
    }
    buf.push(val);
}

fn uptime_str(secs: u64) -> String {
    format!("{}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

/// Pulse character based on tick.
fn pulse(tick: u64) -> &'static str {
    PULSE_CHARS[(tick / 2) as usize % PULSE_CHARS.len()]
}

// ── Main Render ────────────────────────────────────────────────────────

pub fn render_interactive(frame: &mut Frame, state: &InteractiveDashState) {
    let area = frame.area();
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),   // Tab bar
            Constraint::Length(3),   // Hero: model + GPU + status
            Constraint::Min(10),    // Main content
            Constraint::Length(1),   // Status bar
        ])
        .split(area);

    render_tabs(frame, outer[0], state);
    render_hero(frame, outer[1], state);
    match state.mode {
        Mode::Monitor => render_monitor(frame, outer[2], state),
        Mode::Prove => render_prove_mode(frame, outer[2], state),
        Mode::Chat => render_chat_mode(frame, outer[2], state),
        Mode::OnChain => render_onchain_mode(frame, outer[2], state),
    }
    render_status_bar(frame, outer[3], state);
}

// ── Tab Bar ────────────────────────────────────────────────────────────

fn render_tabs(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let titles: Vec<Line> = Mode::titles().iter().map(|t| Line::from(*t)).collect();
    let tabs = Tabs::new(titles)
        .select(state.mode.index())
        .highlight_style(Style::default().fg(T.bg).bg(T.highlight).add_modifier(Modifier::BOLD))
        .style(Style::default().fg(T.muted))
        .divider(Span::styled(" │ ", Style::default().fg(T.border)));
    frame.render_widget(tabs, area);
}

// ── Hero Panel ─────────────────────────────────────────────────────────

fn render_hero(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let model = state.models.get(state.selected_model);
    let model_name = model.map(|m| m.name.as_str()).unwrap_or("none");
    let model_params = model.map(|m| m.params.as_str()).unwrap_or("");

    let phase_color = match state.proving_phase.as_str() {
        "gkr" => T.accent,
        "stark" => T.highlight,
        "on-chain" => T.violet,
        "forward" => T.metric,
        _ => T.muted,
    };

    let p = pulse(state.tick);

    let lines = vec![
        Line::from(vec![
            badge("ObelyZK", T.bg, T.accent),
            Span::styled("  ", Style::default()),
            badge(model_name, T.text, T.violet),
            Span::styled(format!("  {} ", model_params), Style::default().fg(T.muted)),
            Span::styled(format!(" {} layers  {} matmuls",
                model.map(|m| m.layers).unwrap_or(0),
                model.map(|m| m.matmuls).unwrap_or(0),
            ), Style::default().fg(T.muted)),
        ]),
        Line::from(vec![
            badge("GPU", T.bg, T.metric),
            Span::styled(format!("  {} ", state.gpu_name), Style::default().fg(T.text)),
            Span::styled(format!(" {:.0}GB ", state.gpu_memory_gb), Style::default().fg(T.muted)),
            Span::styled(format!(" {:.0}°C ", state.gpu_temp_c), Style::default().fg(
                if state.gpu_temp_c > 80.0 { T.danger } else if state.gpu_temp_c > 60.0 { T.metric } else { T.success }
            )),
            Span::styled("  ", Style::default()),
            badge(&state.proving_phase.to_uppercase(), T.bg, phase_color),
            Span::styled(format!(" {} ", p), Style::default().fg(phase_color)),
            if state.proving_active {
                Span::styled(format!("{:.1}s", state.proving_elapsed_secs), Style::default().fg(T.metric))
            } else {
                Span::styled("", Style::default())
            },
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

// ── Mode: Monitor ──────────────────────────────────────────────────────

fn render_monitor(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),   // Sparklines row
            Constraint::Length(6),   // Layer timings + layer trace
            Constraint::Min(3),     // Stats
        ])
        .split(area);

    // Row 1: Three sparklines side by side
    let spark_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(34), Constraint::Percentage(33), Constraint::Percentage(33)])
        .split(rows[0]);

    // Throughput sparkline
    let tw = spark_cols[0].width.saturating_sub(2) as usize;
    let ts = sparkline_str(&state.throughput_history, tw);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(&ts, Style::default().fg(T.accent))),
            Line::from(vec![
                Span::styled(format!(" {:.1}", state.current_tok_per_sec), Style::default().fg(T.accent).add_modifier(Modifier::BOLD)),
                Span::styled(" tok/s  ", Style::default().fg(T.muted)),
                Span::styled(format!("peak {:.1}", state.peak_tok_per_sec), Style::default().fg(T.metric)),
            ]),
        ]).block(panel("Throughput", T.accent)),
        spark_cols[0],
    );

    // GPU utilization sparkline
    let gw = spark_cols[1].width.saturating_sub(2) as usize;
    let gs = sparkline_str(&state.gpu_util_history, gw);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(&gs, Style::default().fg(T.metric))),
            Line::from(vec![
                Span::styled(format!(" {:.0}%", state.gpu_util_pct), Style::default().fg(T.metric).add_modifier(Modifier::BOLD)),
                Span::styled(" util", Style::default().fg(T.muted)),
            ]),
        ]).block(panel("GPU Load", T.metric)),
        spark_cols[1],
    );

    // Layer timing sparkline
    let lw = spark_cols[2].width.saturating_sub(2) as usize;
    let ls = sparkline_str(&state.layer_time_history, lw);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(&ls, Style::default().fg(T.violet))),
            Line::from(vec![
                Span::styled(format!(" {}ms", state.layer_time_history.last().unwrap_or(&0)), Style::default().fg(T.violet).add_modifier(Modifier::BOLD)),
                Span::styled(" /layer", Style::default().fg(T.muted)),
            ]),
        ]).block(panel("Layer Time", T.violet)),
        spark_cols[2],
    );

    // Row 2: Layer trace + proving info
    let mid_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(rows[1]);

    // Layer trace (recent)
    let trace_items: Vec<ListItem> = state.layer_events.iter().take(4).map(|e| {
        let sym = if e.done { "✓" } else { "▸" };
        let sym_color = if e.done { T.success } else { T.metric };
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {} ", sym), Style::default().fg(sym_color)),
            Span::styled(format!("L{:<3}", e.layer_idx), Style::default().fg(T.muted)),
            Span::styled(format!(" {:<10}", e.layer_type), Style::default().fg(T.text)),
            Span::styled(format!(" {}ms", e.time_ms), Style::default().fg(T.metric)),
        ]))
    }).collect();
    frame.render_widget(
        List::new(trace_items).block(panel("Proof Trace", T.cyan)),
        mid_cols[0],
    );

    // On-chain recent
    let tx_items: Vec<ListItem> = state.on_chain_txs.iter().take(4).map(|tx| {
        let short = if tx.tx_hash.len() > 22 { &tx.tx_hash[..22] } else { &tx.tx_hash };
        ListItem::new(Line::from(vec![
            Span::styled(if tx.verified { " ✓ " } else { " · " }, Style::default().fg(if tx.verified { T.success } else { T.muted })),
            Span::styled(short, Style::default().fg(T.violet)),
            Span::styled(format!(" {} ", tx.model), Style::default().fg(T.muted)),
            Span::styled(format!("{}f", tx.felts), Style::default().fg(T.metric)),
        ]))
    }).collect();
    frame.render_widget(
        List::new(tx_items).block(panel("On-Chain", T.violet)),
        mid_cols[1],
    );

    // Row 3: Stats bar
    let stats_line = Line::from(vec![
        Span::styled(" proven ", Style::default().fg(T.muted)),
        Span::styled(format!("{}", state.total_tokens_proven), Style::default().fg(T.accent).add_modifier(Modifier::BOLD)),
        Span::styled(" tokens  ", Style::default().fg(T.muted)),
        Span::styled(format!("{}", state.total_proofs), Style::default().fg(T.text)),
        Span::styled(" proofs  ", Style::default().fg(T.muted)),
        Span::styled(format!("{}", state.verification_count), Style::default().fg(T.violet).add_modifier(Modifier::BOLD)),
        Span::styled(" verified on-chain  ", Style::default().fg(T.muted)),
        Span::styled(&state.network, Style::default().fg(T.violet)),
    ]);
    frame.render_widget(
        Paragraph::new(stats_line).block(panel("Stats", T.text)),
        rows[2],
    );
}

// ── Mode: Prove ────────────────────────────────────────────────────────

fn render_prove_mode(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // GKR progress
            Constraint::Length(3),   // STARK status
            Constraint::Min(4),     // Proof trace
        ])
        .split(area);

    // GKR progress gauge
    let ratio = if state.proving_total_layers > 0 {
        state.proving_layer as f64 / state.proving_total_layers as f64
    } else { 0.0 };
    let gw = rows[0].width.saturating_sub(4) as usize;
    let mut gauge_spans = vec![Span::styled(" ", Style::default())];
    gauge_spans.extend(gauge_line(ratio, gw, T.accent, T.muted));

    let label = if state.proving_active {
        format!(" GKR {}/{} layers · {}/{} matmuls · {:.1}s ",
            state.proving_layer, state.proving_total_layers,
            state.proving_matmul, state.proving_total_matmuls,
            state.proving_elapsed_secs)
    } else {
        " Press Enter to start proving ".into()
    };

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(gauge_spans),
            Line::from(Span::styled(label, Style::default().fg(if state.proving_active { T.accent } else { T.muted }))),
        ]).block(panel("GKR Proof", T.accent)),
        rows[0],
    );

    // STARK compression
    let stark_lines = if let Some(t) = state.stark_time_secs {
        vec![Line::from(vec![
            badge("STARK", T.bg, T.success),
            Span::styled(format!(" {:.1}s ", t), Style::default().fg(T.metric).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {} felts ", state.stark_felts.unwrap_or(0)), Style::default().fg(T.text)),
            badge("ON-CHAIN READY", T.bg, T.highlight),
        ])]
    } else {
        vec![Line::from(vec![
            Span::styled(format!(" {} ", pulse(state.tick)), Style::default().fg(T.muted)),
            Span::styled(
                if state.proving_active { "Waiting for GKR..." } else { "No active proof" },
                Style::default().fg(T.muted),
            ),
        ])]
    };
    frame.render_widget(
        Paragraph::new(stark_lines).block(panel("Recursive STARK", T.highlight)),
        rows[1],
    );

    // Proof trace
    let trace_items: Vec<ListItem> = state.layer_events.iter().take(15).map(|e| {
        let sym = if e.done { "✓" } else { "▸" };
        let color = if e.done { T.success } else { T.metric };
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {}", sym), Style::default().fg(color)),
            Span::styled(format!(" L{:<4}", e.layer_idx), Style::default().fg(T.muted)),
            Span::styled(format!("{:<12}", e.layer_type), Style::default().fg(T.text)),
            Span::styled(format!(" {:>4}ms", e.time_ms), Style::default().fg(T.metric)),
            Span::styled(format!(" {}", "█".repeat((e.time_ms as usize / 50).min(20))), Style::default().fg(T.accent_soft)),
        ]))
    }).collect();
    frame.render_widget(
        List::new(trace_items).block(panel("Live Trace", T.cyan)),
        rows[2],
    );
}

// ── Mode: Chat ─────────────────────────────────────────────────────────

fn render_chat_mode(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(3)])
        .split(area);

    // Messages
    let items: Vec<ListItem> = state.chat_messages.iter().map(|m| {
        let (label, label_bg) = if m.role == "user" { ("YOU", T.accent) } else { ("AI", T.highlight) };
        let mut spans = vec![
            badge(label, T.bg, label_bg),
            Span::styled(format!(" {}", m.content), Style::default().fg(T.text)),
        ];
        if let Some(ref st) = m.proof_status {
            let sc = if st == "verified" { T.success } else { T.metric };
            spans.push(Span::styled(format!("  [{}]", st), Style::default().fg(sc)));
        }
        if let Some(ref tx) = m.tx_hash {
            let short = if tx.len() > 18 { &tx[..18] } else { tx.as_str() };
            spans.push(Span::styled(format!("  TX:{}", short), Style::default().fg(T.violet)));
        }
        ListItem::new(Line::from(spans))
    }).collect();

    frame.render_widget(
        List::new(items).block(panel("Verifiable Chat", T.accent)),
        rows[0],
    );

    // Input
    let input_spans = vec![
        Span::styled(" › ", Style::default().fg(T.accent).add_modifier(Modifier::BOLD)),
        Span::styled(state.input_buffer.clone(), Style::default().fg(T.text)),
        Span::styled("█", Style::default().fg(T.highlight)),
    ];
    frame.render_widget(
        Paragraph::new(Line::from(input_spans))
            .block(Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(T.highlight))
                .title(Span::styled(" Enter to send · Esc to cancel ", Style::default().fg(T.muted)))),
        rows[1],
    );
}

// ── Mode: On-Chain ─────────────────────────────────────────────────────

fn render_onchain_mode(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(6)])
        .split(area);

    // Contract info
    let contract_short = if state.contract.len() > 40 {
        format!("{}...{}", &state.contract[..20], &state.contract[state.contract.len()-8..])
    } else { state.contract.clone() };

    frame.render_widget(
        Paragraph::new(vec![
            Line::from(vec![
                badge("CONTRACT", T.bg, T.violet),
                Span::styled(format!("  {}", contract_short), Style::default().fg(T.text)),
            ]),
            Line::from(vec![
                badge(&state.network, T.bg, T.cyan),
                Span::styled(format!("  {} verifications", state.verification_count), Style::default().fg(T.accent).add_modifier(Modifier::BOLD)),
            ]),
        ]).block(panel("Starknet", T.violet)),
        rows[0],
    );

    // TX feed
    let tx_items: Vec<ListItem> = state.on_chain_txs.iter().rev().map(|tx| {
        let short = if tx.tx_hash.len() > 24 { &tx.tx_hash[..24] } else { &tx.tx_hash };
        ListItem::new(Line::from(vec![
            badge(if tx.verified { "VERIFIED" } else { "PENDING" }, T.bg, if tx.verified { T.success } else { T.metric }),
            Span::styled(format!("  {}", short), Style::default().fg(T.violet)),
            Span::styled(format!("  {} ", tx.model), Style::default().fg(T.text)),
            Span::styled(format!(" {} felts", tx.felts), Style::default().fg(T.metric)),
        ]))
    }).collect();
    frame.render_widget(
        List::new(tx_items).block(panel("Verified Proofs", T.success)),
        rows[1],
    );
}

// ── Status Bar ─────────────────────────────────────────────────────────

fn render_status_bar(frame: &mut Frame, area: Rect, state: &InteractiveDashState) {
    let status_color = if state.proving_active { T.metric } else { T.success };
    let status_text = if state.proving_active { "PROVING" } else { "READY" };

    let line = Line::from(vec![
        badge("ObelyZK", T.bg, T.accent),
        Span::styled(format!(" {} ", status_text), Style::default().fg(status_color).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" {} ", uptime_str(state.uptime_secs)), Style::default().fg(T.metric)),
        Span::styled(format!(" {} ", state.network), Style::default().fg(T.violet)),
        Span::styled(format!(" {} TXs ", state.verification_count), Style::default().fg(T.muted)),
        Span::styled("  ", Style::default()),
        Span::styled("1-4", Style::default().fg(T.accent)),
        Span::styled(" mode ", Style::default().fg(T.muted)),
        Span::styled("q", Style::default().fg(T.accent)),
        Span::styled(" quit ", Style::default().fg(T.muted)),
        Span::styled("Enter", Style::default().fg(T.accent)),
        Span::styled(" prove ", Style::default().fg(T.muted)),
        Span::styled("Tab", Style::default().fg(T.accent)),
        Span::styled(" model", Style::default().fg(T.muted)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}
