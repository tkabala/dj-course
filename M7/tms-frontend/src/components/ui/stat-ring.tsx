import React from 'react';
import { cn } from '@/lib/tailwind/utils';

export interface StatRingProps {
  /** Progress value shown by the ring, 0-100 */
  value: number;
  /** Text rendered in the center of the ring (e.g. "98%", "72") */
  displayValue: string;
  /** Label rendered below the ring */
  label: string;
  variant?: 'blue' | 'green' | 'purple' | 'orange' | 'red';
  /** Diameter in px */
  size?: number;
  icon?: React.ReactNode;
  className?: string;
}

const variantStyles: Record<NonNullable<StatRingProps['variant']>, { stroke: string; track: string; text: string }> = {
  blue: { stroke: 'stroke-blue-500', track: 'stroke-blue-100', text: 'text-blue-600' },
  green: { stroke: 'stroke-green-500', track: 'stroke-green-100', text: 'text-green-600' },
  purple: { stroke: 'stroke-purple-500', track: 'stroke-purple-100', text: 'text-purple-600' },
  orange: { stroke: 'stroke-orange-500', track: 'stroke-orange-100', text: 'text-orange-600' },
  red: { stroke: 'stroke-red-500', track: 'stroke-red-100', text: 'text-red-600' },
};

export const StatRing: React.FC<StatRingProps> = ({
  value,
  displayValue,
  label,
  variant = 'blue',
  size = 96,
  icon,
  className,
}) => {
  const styles = variantStyles[variant];
  const strokeWidth = size * 0.09;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(100, Math.max(0, value));
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className={cn('flex flex-col items-center gap-2', className)}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            fill="none"
            className={styles.track}
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            strokeWidth={strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={cn(styles.stroke, 'transition-all duration-700 ease-out')}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          {icon && <div className={cn('mb-0.5', styles.text)}>{icon}</div>}
          <span className={cn('font-bold leading-none', styles.text)} style={{ fontSize: size * 0.19 }}>
            {displayValue}
          </span>
        </div>
      </div>
      <span className="text-sm text-gray-600 text-center">{label}</span>
    </div>
  );
};
