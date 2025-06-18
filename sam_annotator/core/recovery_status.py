#!/usr/bin/env python3
"""Recovery status indicators and reporting system for SAM Annotator"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta


class RecoveryStatus(Enum):
    """Current recovery system status"""
    HEALTHY = "healthy"
    MONITORING = "monitoring"
    RECOVERING = "recovering"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class RecoveryEvent:
    """Individual recovery event record"""
    timestamp: datetime
    operation: str
    stage: str
    success: bool
    recovery_time: float
    memory_recovered: float
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'operation': self.operation,
            'stage': self.stage,
            'success': self.success,
            'recovery_time_ms': round(self.recovery_time * 1000, 2),
            'memory_recovered_mb': round(self.memory_recovered, 2),
            'message': self.message
        }


class RecoveryStatusReporter:
    """System for tracking and reporting recovery status to users"""
    
    def __init__(self, max_events: int = 100):
        self.max_events = max_events
        self.events: List[RecoveryEvent] = []
        self.current_status = RecoveryStatus.HEALTHY
        self.status_start_time = datetime.now()
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'avg_recovery_time': 0.0,
            'total_memory_recovered': 0.0,
            'recent_success_rate': 1.0,  # Last 10 attempts
            'uptime_hours': 0.0
        }
        
        # Status thresholds
        self.thresholds = {
            'monitoring_failure_rate': 0.1,    # 10% failure rate triggers monitoring
            'degraded_failure_rate': 0.3,      # 30% failure rate = degraded
            'critical_failure_rate': 0.6,      # 60% failure rate = critical
            'recovery_timeout': 10.0,          # 10 seconds max recovery time
        }

    def record_recovery_event(self, operation: str, stage: str, success: bool, 
                            recovery_time: float, memory_recovered: float, message: str):
        """Record a recovery event and update status"""
        
        # Create event record
        event = RecoveryEvent(
            timestamp=datetime.now(),
            operation=operation,
            stage=stage,
            success=success,
            recovery_time=recovery_time,
            memory_recovered=memory_recovered,
            message=message
        )
        
        # Add to events list (maintain max size)
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
        
        # Update metrics
        self._update_metrics(event)
        
        # Update status based on recent performance
        self._update_status()
        
        # Log important events
        if not success:
            self.logger.warning(f"Recovery failed: {operation} - {message}")
        elif recovery_time > self.thresholds['recovery_timeout']:
            self.logger.info(f"Slow recovery: {operation} took {recovery_time:.2f}s")

    def _update_metrics(self, event: RecoveryEvent):
        """Update performance metrics with new event"""
        self.metrics['total_recoveries'] += 1
        
        if event.success:
            self.metrics['successful_recoveries'] += 1
        else:
            self.metrics['failed_recoveries'] += 1
        
        # Update average recovery time (running average)
        total_time = self.metrics['avg_recovery_time'] * (self.metrics['total_recoveries'] - 1)
        self.metrics['avg_recovery_time'] = (total_time + event.recovery_time) / self.metrics['total_recoveries']
        
        # Update total memory recovered
        self.metrics['total_memory_recovered'] += event.memory_recovered
        
        # Calculate recent success rate (last 10 events)
        recent_events = self.events[-10:]
        if recent_events:
            recent_successes = sum(1 for e in recent_events if e.success)
            self.metrics['recent_success_rate'] = recent_successes / len(recent_events)
        
        # Update uptime
        uptime_delta = datetime.now() - self.status_start_time
        self.metrics['uptime_hours'] = uptime_delta.total_seconds() / 3600

    def _update_status(self):
        """Update system status based on recent performance"""
        failure_rate = 1.0 - self.metrics['recent_success_rate']
        
        # Determine new status
        new_status = RecoveryStatus.HEALTHY
        
        if failure_rate >= self.thresholds['critical_failure_rate']:
            new_status = RecoveryStatus.CRITICAL
        elif failure_rate >= self.thresholds['degraded_failure_rate']:
            new_status = RecoveryStatus.DEGRADED
        elif failure_rate >= self.thresholds['monitoring_failure_rate']:
            new_status = RecoveryStatus.MONITORING
        
        # Check for active recovery
        recent_events = self.events[-5:]  # Last 5 events
        if recent_events and any(not e.success for e in recent_events):
            if new_status == RecoveryStatus.HEALTHY:
                new_status = RecoveryStatus.MONITORING
        
        # Update status if changed
        if new_status != self.current_status:
            old_status = self.current_status
            self.current_status = new_status
            self.status_start_time = datetime.now()
            
            self.logger.info(f"Recovery status changed: {old_status.value} → {new_status.value}")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary for UI display"""
        
        # Get recent events (last 10)
        recent_events = self.events[-10:]
        
        # Calculate time since last event
        time_since_last = None
        if self.events:
            time_since_last = (datetime.now() - self.events[-1].timestamp).total_seconds()
        
        # Get status color and message
        status_info = self._get_status_display_info()
        
        return {
            'status': {
                'current': self.current_status.value,
                'color': status_info['color'],
                'message': status_info['message'],
                'since': self.status_start_time.isoformat()
            },
            'metrics': self.metrics.copy(),
            'recent_events': [event.to_dict() for event in recent_events],
            'system_health': {
                'last_event_seconds_ago': time_since_last,
                'recovery_system_active': len(self.events) > 0,
                'needs_attention': self.current_status in [RecoveryStatus.DEGRADED, RecoveryStatus.CRITICAL]
            },
            'recommendations': self._get_recommendations()
        }

    def _get_status_display_info(self) -> Dict[str, str]:
        """Get display information for current status"""
        status_map = {
            RecoveryStatus.HEALTHY: {
                'color': 'green',
                'message': 'Memory recovery system operating normally'
            },
            RecoveryStatus.MONITORING: {
                'color': 'yellow',
                'message': 'Monitoring memory usage - some recovery attempts detected'
            },
            RecoveryStatus.RECOVERING: {
                'color': 'orange', 
                'message': 'Active memory recovery in progress'
            },
            RecoveryStatus.DEGRADED: {
                'color': 'orange',
                'message': 'Memory recovery system under stress - performance may be affected'
            },
            RecoveryStatus.CRITICAL: {
                'color': 'red',
                'message': 'Critical memory issues - immediate attention recommended'
            }
        }
        
        return status_map.get(self.current_status, {
            'color': 'gray',
            'message': 'Recovery status unknown'
        })

    def _get_recommendations(self) -> List[str]:
        """Get actionable recommendations based on current status"""
        recommendations = []
        
        if self.current_status == RecoveryStatus.CRITICAL:
            recommendations.extend([
                "Reduce image resolution or batch size",
                "Close other GPU-intensive applications",
                "Consider restarting the application",
                "Check system memory and GPU memory availability"
            ])
        elif self.current_status == RecoveryStatus.DEGRADED:
            recommendations.extend([
                "Monitor memory usage closely",
                "Consider reducing processing complexity",
                "Check for memory leaks in long-running sessions"
            ])
        elif self.current_status == RecoveryStatus.MONITORING:
            recommendations.extend([
                "Memory usage is elevated but manageable",
                "Consider clearing caches if performance degrades"
            ])
        
        # Add specific recommendations based on metrics
        if self.metrics['avg_recovery_time'] > 5.0:
            recommendations.append("Recovery times are high - consider increasing GPU memory allocation")
        
        if self.metrics['total_recoveries'] > 50:
            recommendations.append("Frequent recoveries detected - review memory configuration")
        
        return recommendations

    def get_simple_status(self) -> str:
        """Get simple status string for logging"""
        return f"Recovery: {self.current_status.value} | Success: {self.metrics['recent_success_rate']:.1%} | Events: {len(self.events)}"

    def clear_old_events(self, hours: int = 24):
        """Clear events older than specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        self.events = [event for event in self.events if event.timestamp > cutoff_time]
        
        self.logger.info(f"Cleared recovery events older than {hours} hours")

    def export_events(self) -> List[Dict[str, Any]]:
        """Export all events for external analysis"""
        return [event.to_dict() for event in self.events]

    def get_health_check(self) -> Dict[str, Any]:
        """Get health check information for monitoring systems"""
        return {
            'status': self.current_status.value,
            'healthy': self.current_status in [RecoveryStatus.HEALTHY, RecoveryStatus.MONITORING],
            'success_rate': self.metrics['recent_success_rate'],
            'total_recoveries': self.metrics['total_recoveries'],
            'uptime_hours': round(self.metrics['uptime_hours'], 2),
            'last_event': self.events[-1].to_dict() if self.events else None
        }


# Global recovery status reporter (singleton-like)
_recovery_reporter = None

def get_recovery_reporter() -> RecoveryStatusReporter:
    """Get global recovery status reporter"""
    global _recovery_reporter
    if _recovery_reporter is None:
        _recovery_reporter = RecoveryStatusReporter()
    return _recovery_reporter 