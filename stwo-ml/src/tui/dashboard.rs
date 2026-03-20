//! Proof dashboard — beautiful terminal visualization of the proving pipeline.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Gauge, List, ListItem, Padding, Paragraph, Wrap},
    Frame,
};

/// State for the proof dashboard.
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub model_name: String,
    pub model_params: String,
    pub model_layers: u32,
    pub num_turns: usize,
    pub tokens_in: usize,
    pub tokens_out: usize,

    // Pipeline progress
    pub step: PipelineStep,
    pub capture_progress: f64,   // 0.0 - 1.0
    pub prove_progress: f64,
    pub recursive_progress: f64,
    pub verify_progress: f64,

    // Timing
    pub capture_time: Option<f64>,
    pub prove_time: Option<f64>,
    pub recursive_time: Option<f64>,

    // Commitments
    pub weight_commitment: Option<String>,
    pub io_root: Option<String>,
    pub report_hash: Option<String>,

    // On-chain
    pub contract: String,
    pub network: String,
    pub verification_count: Option<u64>,

    // Tamper tests
    pub tamper_io: Option<bool>,
    pub tamper_weight: Option<bool>,
    pub tamper_output: Option<bool>,

    // Conversation
    pub turns: Vec<(String, String)>, // (user, ai)

    // Log messages
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PipelineStep {
    Idle,
    Capture,
    Prove,
    Recursive,
    Verify,
    Complete,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            model_name: "qwen2-0.5b".into(),
            model_params: "247,726,080".into(),
            model_layers: 169,
            num_turns: 0,
            tokens_in: 0,
            tokens_out: 0,
            step: PipelineStep::Idle,
            capture_progress: 0.0,
            prove_progress: 0.0,
            recursive_progress: 0.0,
            verify_progress: 0.0,
            capture_time: None,
            prove_time: None,
            recursive_time: None,
            weight_commitment: None,
            io_root: None,
            report_hash: None,
            contract: "0x0121d1e9882967e03399f153d57fc208f3d9bce69adc48d9e12d424502a8c005".into(),
            network: "Starknet Sepolia".into(),
            verification_count: None,
            tamper_io: None,
            tamper_weight: None,
            tamper_output: None,
            turns: Vec::new(),
            logs: Vec::new(),
        }
    }
}

/// Render the proof dashboard.
pub fn render(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();

    // Main layout: header + body + footer
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),   // Body
            Constraint::Length(3), // Footer
        ])
        .split(area);

    render_header(frame, main_layout[0], state);
    render_body(frame, main_layout[1], state);
    render_footer(frame, main_layout[2], state);
}

fn render_header(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let title = Line::from(vec![
        Span::styled(" OBELYSK ", Style::default().fg(Color::Black).bg(Color::Cyan).bold()),
        Span::raw("  "),
        Span::styled("Verifiable ML Inference", Style::default().fg(Color::White).bold()),
        Span::raw("  "),
        Span::styled(
            format!("{} • {} params • {} layers",
                state.model_name, state.model_params, state.model_layers),
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let header = Paragraph::new(title)
        .block(Block::default().borders(Borders::BOTTOM).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(header, area);
}

fn render_body(frame: &mut Frame, area: Rect, state: &DashboardState) {
    // Split body: left (pipeline + progress) | right (commitments + conversation)
    let body_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55),
            Constraint::Percentage(45),
        ])
        .split(area);

    render_pipeline(frame, body_layout[0], state);
    render_details(frame, body_layout[1], state);
}

fn render_pipeline(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Step 1
            Constraint::Length(3), // Step 2
            Constraint::Length(3), // Step 3
            Constraint::Length(3), // Step 4
            Constraint::Min(3),   // Logs
        ])
        .split(area);

    // Step 1: Capture
    render_step(frame, layout[0], "1", "Capture", "M31 forward passes",
        state.capture_progress, state.capture_time,
        state.step == PipelineStep::Capture || state.step.as_u8() > PipelineStep::Capture.as_u8());

    // Step 2: Prove
    render_step(frame, layout[1], "2", "Prove", "GKR sumcheck (96 matmuls)",
        state.prove_progress, state.prove_time,
        state.step == PipelineStep::Prove || state.step.as_u8() > PipelineStep::Prove.as_u8());

    // Step 3: Recursive
    render_step(frame, layout[2], "3", "Recursive", "STARK compression",
        state.recursive_progress, state.recursive_time,
        state.step == PipelineStep::Recursive || state.step.as_u8() > PipelineStep::Recursive.as_u8());

    // Step 4: Verify
    render_step(frame, layout[3], "4", "Verify", "Self-verify + on-chain",
        state.verify_progress, None,
        state.step == PipelineStep::Verify || state.step == PipelineStep::Complete);

    // Logs
    let log_items: Vec<ListItem> = state.logs.iter().rev().take(8).map(|l| {
        ListItem::new(Span::styled(l.clone(), Style::default().fg(Color::DarkGray)))
    }).collect();
    let logs = List::new(log_items)
        .block(Block::default().title(" Log ").borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(logs, layout[4]);
}

fn render_step(
    frame: &mut Frame,
    area: Rect,
    num: &str,
    name: &str,
    desc: &str,
    progress: f64,
    time: Option<f64>,
    active: bool,
) {
    let color = if progress >= 1.0 {
        Color::Green
    } else if active {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    let check = if progress >= 1.0 { "✓" } else if active { "▸" } else { " " };
    let time_str = time.map(|t| format!(" ({:.1}s)", t)).unwrap_or_default();

    let label = format!(" {check} [{num}] {name}: {desc}{time_str} ");
    let gauge = Gauge::default()
        .block(Block::default())
        .gauge_style(Style::default().fg(color))
        .ratio(progress.clamp(0.0, 1.0))
        .label(Span::styled(label, Style::default().fg(Color::White)));

    frame.render_widget(gauge, area);
}

fn render_details(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),  // Commitments
            Constraint::Length(6),  // Tamper tests
            Constraint::Min(5),    // Conversation
        ])
        .split(area);

    // Commitments
    let mut commit_lines = vec![
        Line::from(vec![
            Span::styled("Weight: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                state.weight_commitment.as_deref().unwrap_or("pending..."),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("IO:     ", Style::default().fg(Color::Yellow)),
            Span::styled(
                state.io_root.as_deref().unwrap_or("pending..."),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Report: ", Style::default().fg(Color::Yellow)),
            Span::styled(
                state.report_hash.as_deref().unwrap_or("pending..."),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Contract: ", Style::default().fg(Color::Cyan)),
            Span::styled(&state.contract[..20], Style::default().fg(Color::DarkGray)),
            Span::styled("...", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Network:  ", Style::default().fg(Color::Cyan)),
            Span::styled(&state.network, Style::default().fg(Color::White)),
        ]),
    ];

    if let Some(count) = state.verification_count {
        commit_lines.push(Line::from(vec![
            Span::styled("Verified: ", Style::default().fg(Color::Green)),
            Span::styled(format!("{count} proofs on-chain"), Style::default().fg(Color::White).bold()),
        ]));
    }

    let commits = Paragraph::new(commit_lines)
        .block(Block::default().title(" Commitments ").borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow))
            .padding(Padding::horizontal(1)));
    frame.render_widget(commits, layout[0]);

    // Tamper tests
    let tamper_lines = vec![
        tamper_line("IO commitment", state.tamper_io),
        tamper_line("Weight commitment", state.tamper_weight),
        tamper_line("Inference output", state.tamper_output),
        Line::from(Span::styled("  56 adversarial tests in suite", Style::default().fg(Color::DarkGray))),
    ];

    let tamper = Paragraph::new(tamper_lines)
        .block(Block::default().title(" Tamper Detection ").borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Red))
            .padding(Padding::horizontal(1)));
    frame.render_widget(tamper, layout[1]);

    // Conversation
    let mut conv_lines: Vec<Line> = Vec::new();
    for (user, ai) in state.turns.iter().rev().take(4) {
        conv_lines.push(Line::from(vec![
            Span::styled("You: ", Style::default().fg(Color::Green).bold()),
            Span::raw(user),
        ]));
        let ai_short: String = ai.chars().take(50).collect();
        conv_lines.push(Line::from(vec![
            Span::styled("AI:  ", Style::default().fg(Color::Cyan).bold()),
            Span::styled(ai_short, Style::default().fg(Color::DarkGray)),
        ]));
        conv_lines.push(Line::raw(""));
    }

    let conv = Paragraph::new(conv_lines)
        .block(Block::default().title(" Conversation ").borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Green))
            .padding(Padding::horizontal(1)))
        .wrap(Wrap { trim: true });
    frame.render_widget(conv, layout[2]);
}

fn tamper_line(name: &str, result: Option<bool>) -> Line<'static> {
    match result {
        Some(true) => Line::from(vec![
            Span::styled("  ✓ ", Style::default().fg(Color::Green)),
            Span::styled(format!("{name} tampered"), Style::default().fg(Color::White)),
            Span::styled(" → REJECTED", Style::default().fg(Color::Green).bold()),
        ]),
        Some(false) => Line::from(vec![
            Span::styled("  ✗ ", Style::default().fg(Color::Red)),
            Span::styled(format!("{name} tampered"), Style::default().fg(Color::White)),
            Span::styled(" → NOT DETECTED", Style::default().fg(Color::Red).bold()),
        ]),
        None => Line::from(vec![
            Span::styled("  · ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{name}"), Style::default().fg(Color::DarkGray)),
            Span::styled(" → pending", Style::default().fg(Color::DarkGray)),
        ]),
    }
}

fn render_footer(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let status = match state.step {
        PipelineStep::Idle => "Ready",
        PipelineStep::Capture => "Capturing forward passes...",
        PipelineStep::Prove => "Generating GKR proofs...",
        PipelineStep::Recursive => "Compressing with recursive STARK...",
        PipelineStep::Verify => "Verifying on-chain...",
        PipelineStep::Complete => "Verification complete ✓",
    };

    let footer = Paragraph::new(Line::from(vec![
        Span::styled(" STWO ML ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw("  "),
        Span::styled(status, Style::default().fg(Color::White)),
        Span::raw("  "),
        Span::styled(
            format!("{} turns • {} in • {} out",
                state.num_turns, state.tokens_in, state.tokens_out),
            Style::default().fg(Color::DarkGray),
        ),
    ]))
    .block(Block::default().borders(Borders::TOP).border_style(Style::default().fg(Color::DarkGray)));
    frame.render_widget(footer, area);
}

// Implement PartialOrd for PipelineStep comparison
impl PipelineStep {
    fn as_u8(&self) -> u8 {
        match self {
            PipelineStep::Idle => 0,
            PipelineStep::Capture => 1,
            PipelineStep::Prove => 2,
            PipelineStep::Recursive => 3,
            PipelineStep::Verify => 4,
            PipelineStep::Complete => 5,
        }
    }
}
