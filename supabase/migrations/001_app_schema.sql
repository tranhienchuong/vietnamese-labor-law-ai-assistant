create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text not null unique,
  full_name text,
  avatar_url text,
  role text not null default 'user' check (role in ('user', 'admin')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  title text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_message_at timestamptz
);

create index if not exists idx_conversations_user_updated
  on public.conversations(user_id, updated_at desc);

create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content text not null,
  citations_json jsonb,
  metadata_json jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_messages_conversation_created
  on public.messages(conversation_id, created_at asc);

create table if not exists public.chat_traces (
  id uuid primary key default gen_random_uuid(),
  request_id text,
  user_id uuid not null references public.profiles(id) on delete cascade,
  conversation_id uuid references public.conversations(id) on delete set null,
  message_id uuid references public.messages(id) on delete set null,
  question text not null,
  provider text,
  model text,
  retrieve_only boolean not null default false,
  insufficient_context boolean not null default false,
  latency_ms integer,
  retrieval_latency_ms integer,
  generation_latency_ms integer,
  intent_json jsonb,
  retrieved_hits_json jsonb,
  selected_contexts_json jsonb,
  citations_json jsonb,
  error text,
  created_at timestamptz not null default now()
);

create index if not exists idx_chat_traces_created_at
  on public.chat_traces(created_at desc);

create index if not exists idx_chat_traces_user_created
  on public.chat_traces(user_id, created_at desc);

create index if not exists idx_chat_traces_conversation
  on public.chat_traces(conversation_id);
