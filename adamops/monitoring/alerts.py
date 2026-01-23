"""
AdamOps Alerts Module

Alerting for model performance degradation.
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from enum import Enum

from adamops.utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """Represents an alert."""
    
    def __init__(self, name: str, message: str, severity: AlertSeverity,
                 metadata: Optional[Dict] = None):
        self.name = name
        self.message = message
        self.severity = severity
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AlertRule:
    """Defines an alert rule."""
    
    def __init__(self, name: str, condition: Callable[[Dict], bool],
                 message_template: str, severity: AlertSeverity = AlertSeverity.WARNING):
        self.name = name
        self.condition = condition
        self.message_template = message_template
        self.severity = severity
    
    def check(self, data: Dict) -> Optional[Alert]:
        """Check if alert should be triggered."""
        if self.condition(data):
            message = self.message_template.format(**data)
            return Alert(self.name, message, self.severity, data)
        return None


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.handlers: List[Callable[[Alert], None]] = []
        self.alert_history: List[Alert] = []
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler (callback)."""
        self.handlers.append(handler)
    
    def check(self, data: Dict) -> List[Alert]:
        """Check all rules against data."""
        triggered = []
        
        for rule in self.rules:
            alert = rule.check(data)
            if alert:
                triggered.append(alert)
                self._handle_alert(alert)
        
        return triggered
    
    def _handle_alert(self, alert: Alert):
        """Handle triggered alert."""
        self.alert_history.append(alert)
        
        # Log alert
        log_method = logger.warning if alert.severity == AlertSeverity.WARNING else \
                     logger.critical if alert.severity == AlertSeverity.CRITICAL else logger.info
        log_method(f"[{alert.severity.value.upper()}] {alert.name}: {alert.message}")
        
        # Call handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_history(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history."""
        if severity:
            return [a for a in self.alert_history if a.severity == severity]
        return self.alert_history


# Default alert rules
def accuracy_drop_rule(threshold: float = 0.1) -> AlertRule:
    """Alert rule for accuracy drop."""
    return AlertRule(
        name="accuracy_drop",
        condition=lambda d: d.get("accuracy_change", 0) < -threshold,
        message_template="Accuracy dropped by {accuracy_change:.1%}",
        severity=AlertSeverity.WARNING,
    )


def drift_detected_rule() -> AlertRule:
    """Alert rule for drift detection."""
    return AlertRule(
        name="drift_detected",
        condition=lambda d: d.get("drift_detected", False),
        message_template="Data drift detected in {drifted_columns} columns",
        severity=AlertSeverity.WARNING,
    )


def high_latency_rule(threshold_ms: float = 1000) -> AlertRule:
    """Alert rule for high latency."""
    return AlertRule(
        name="high_latency",
        condition=lambda d: d.get("latency_ms", 0) > threshold_ms,
        message_template="High latency detected: {latency_ms:.0f}ms",
        severity=AlertSeverity.WARNING,
    )


# Notification handlers
def console_handler(alert: Alert):
    """Print alert to console."""
    print(f"[ALERT] {alert.severity.value.upper()}: {alert.message}")


def email_handler(recipients: List[str], smtp_config: Dict):
    """Create email alert handler."""
    def handler(alert: Alert):
        try:
            import smtplib
            from email.message import EmailMessage
            
            msg = EmailMessage()
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
            msg['From'] = smtp_config.get('from', 'alerts@adamops.local')
            msg['To'] = ', '.join(recipients)
            msg.set_content(f"{alert.message}\n\nTimestamp: {alert.timestamp}")
            
            with smtplib.SMTP(smtp_config['host'], smtp_config.get('port', 587)) as s:
                if smtp_config.get('use_tls'):
                    s.starttls()
                if 'username' in smtp_config:
                    s.login(smtp_config['username'], smtp_config['password'])
                s.send_message(msg)
                
            logger.info(f"Sent email alert to {recipients}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    return handler


def create_alert_manager(
    rules: Optional[List[str]] = None,
    handlers: Optional[List[str]] = None
) -> AlertManager:
    """Create alert manager with default rules."""
    manager = AlertManager()
    
    default_rules = {
        "accuracy": accuracy_drop_rule(),
        "drift": drift_detected_rule(),
        "latency": high_latency_rule(),
    }
    
    rules = rules or list(default_rules.keys())
    for rule_name in rules:
        if rule_name in default_rules:
            manager.add_rule(default_rules[rule_name])
    
    handlers = handlers or ["console"]
    for handler_name in handlers:
        if handler_name == "console":
            manager.add_handler(console_handler)
    
    return manager
